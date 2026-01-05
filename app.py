import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from functools import lru_cache
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="GEX Master Pro",
    page_icon="⏳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- STYLING ---
st.markdown("""
<style>
    .stApp {background-color: #0E1117;}
    div.stButton > button {
        width: 100%;
        background-color: #2962FF;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {background-color: #0039CB;}
    [data-testid="stMetricValue"] {font-size: 1.2rem;}
</style>
""", unsafe_allow_html=True)

# --- CALCULATEUR BLACK-SCHOLES AMÉLIORÉ ---
class GreeksCalculator:
    def __init__(self, risk_free_rate=0.0):
        self.r = risk_free_rate

    def is_last_friday_of_month(self, date):
        """Détection robuste du dernier vendredi"""
        if date.month == 12:
            next_month = date.replace(year=date.year + 1, month=1, day=1)
        else:
            next_month = date.replace(month=date.month + 1, day=1)
        
        last_day = next_month - timedelta(days=1)
        days_to_friday = (last_day.weekday() - 4) % 7
        last_friday = last_day - timedelta(days=days_to_friday)
        
        return date.date() == last_friday.date()

    def calculate(self, contract_data):
        try:
            parts = contract_data['instrument_name'].split('-')
            if len(parts) < 4: 
                return None  # ✅ CORRECTION
            
            date_str = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            expiry = datetime.strptime(date_str, "%d%b%y")
        except:
            return None  # ✅ CORRECTION

        S = contract_data.get('underlying_price', 0)
        K = strike
        sigma = contract_data.get('mark_iv', 0) / 100.0
        
        if S == 0 or sigma == 0 or sigma < 0.01:  # ✅ CORRECTION: Protection IV
            return None

        now = datetime.now()
        T = (expiry - now).total_seconds() / (365 * 24 * 3600)
        
        # ✅ CORRECTION: Validation stricte
        if T <= 0 or T > 5 or T < 1/365:
            return None

        days_to_expiry = (expiry - now).days
        weekday = expiry.weekday()
        month = expiry.month
        
        # ✅ AMÉLIORATION: Détection robuste
        is_monthly = (weekday == 4 and self.is_last_friday_of_month(expiry))
        is_quarterly = is_monthly and (month in [3, 6, 9, 12])

        # Black-Scholes
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # ✅ CORRECTION: Validation résultat
            if np.isnan(gamma) or np.isinf(gamma):
                return None
                
        except:
            return None
        
        contract_data['greeks'] = {"gamma": round(gamma, 5)}
        contract_data['dte_days'] = days_to_expiry
        contract_data['weekday'] = weekday
        contract_data['is_quarterly'] = is_quarterly
        contract_data['is_monthly'] = is_monthly
        contract_data['expiry_date'] = expiry
        
        return contract_data

# --- API AVEC CACHE ---
@lru_cache(maxsize=1)
def get_deribit_data_cached(currency, timestamp):
    """Cache valide 1 minute"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url_spot = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={currency.lower()}_usd"
        spot_res = requests.get(url_spot, headers=headers).json()
        spot = spot_res['result']['index_price']
        
        url_book = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        params = {'currency': currency, 'kind': 'option'}
        book_res = requests.get(url_book, params=params, headers=headers).json()
        data = book_res['result']
        return spot, data
    except Exception as e:
        st.error(f"Erreur API: {e}")
        return None, None

def get_deribit_data(currency='BTC'):
    """Wrapper avec cache par minute"""
    current_minute = int(time.time() / 60)
    return get_deribit_data_cached(currency, current_minute)

# --- ANALYSEUR EXPIRATIONS (Inchangé) ---
def analyze_upcoming_expirations(data):
    expirations = {}
    now = datetime.now()
    
    for entry in data:
        try:
            parts = entry['instrument_name'].split('-')
            date_str = parts[1]
            expiry = datetime.strptime(date_str, "%d%b%y")
            days_left = (expiry - now).days
            
            if days_left < 0: continue
            
            date_key = expiry.strftime("%d %b %Y")
            weekday = expiry.weekday()
            day = expiry.day
            month = expiry.month
            is_monthly = (day > 21 and weekday == 4)
            is_quart = is_monthly and (month in [3, 6, 9, 12])
            
            if is_quart and days_left not in expirations:
                expirations[days_left] = {"date": date_key, "type": "👑 Quarterly"}
            elif is_monthly and days_left not in expirations:
                expirations[days_left] = {"date": date_key, "type": "🏆 Monthly"}
        except:
            continue
    
    sorted_days = sorted(expirations.keys())
    return sorted_days, expirations

# --- PROCESSING AMÉLIORÉ ---
def process_gex(spot, data, dte_limit, only_fridays, use_weighting, w_quart, w_month, w_week):
    calculator = GreeksCalculator()
    strikes = {}
    missed_quarterly_dtes = []
    
    for entry in data:
        contract = calculator.calculate(entry)
        
        if contract is None:  # ✅ CORRECTION: Skip invalides
            continue
            
        instr = contract.get('instrument_name', 'UNKNOWN')
        dte = contract.get('dte_days', 9999)
        is_quart = contract.get('is_quarterly', False)
        is_month = contract.get('is_monthly', False)
        weekday = contract.get('weekday', -1)
        oi = contract.get('open_interest', 0)
        greeks = contract.get('greeks')
        
        if dte > dte_limit:
            if is_quart: 
                missed_quarterly_dtes.append(dte)
            continue
        
        if only_fridays and weekday != 4: 
            continue
        if oi == 0: 
            continue
        if not greeks: 
            continue
        
        try:
            parts = instr.split('-')
            if len(parts) < 4: 
                continue
            strike = float(parts[2])
            opt_type = parts[3]
            gamma = greeks.get('gamma', 0) or 0
            
            weight = 1.0
            if use_weighting:
                if is_quart: weight = w_quart
                elif is_month: weight = w_month
                else: weight = w_week
            
            gex_val = ((gamma * oi * (spot ** 2) / 100) / 1_000_000) * weight
            
            if strike not in strikes: 
                strikes[strike] = {'total_gex': 0}
            if opt_type == 'C': 
                strikes[strike]['total_gex'] += gex_val
            else: 
                strikes[strike]['total_gex'] -= gex_val
        except:
            continue
    
    # ✅ AMÉLIORATION: Alerte plus intelligente
    warnings = []
    if missed_quarterly_dtes:
        next_missed_q = min(missed_quarterly_dtes)
        if next_missed_q < (dte_limit * 2):  # Seuil logique
            warnings.append(
                f"⚠️ QUARTERLY PROCHE IGNORÉE : Dans {next_missed_q} jours "
                f"(horizon: {dte_limit}j). Augmentez à {next_missed_q + 10}j."
            )
    
    if not strikes:
        return pd.DataFrame(), spot, spot, spot, warnings, 0
    
    df = pd.DataFrame.from_dict(strikes, orient='index')
    df.index.name = 'Strike'
    df = df.sort_index()
    
    # ✅ AMÉLIORATION: Call/Put Walls filtrés
    relevant_df = df[(df.index > spot * 0.7) & (df.index < spot * 1.3)]
    if not relevant_df.empty:
        call_wall = relevant_df['total_gex'].idxmax()
        put_wall = relevant_df['total_gex'].idxmin()
    else:
        call_wall = df['total_gex'].idxmax()
        put_wall = df['total_gex'].idxmin()
    
    # Zero Gamma avec range adaptatif (simplifié)
    subset = df[(df.index > spot * 0.85) & (df.index < spot * 1.15)]
    
    if subset.empty:
        subset = df[(df.index > spot * 0.5) & (df.index < spot * 2.0)]
    
    neg_gex = subset[subset['total_gex'] < 0]
    pos_gex = subset[subset['total_gex'] > 0]
    zero_gamma = spot
    
    if not neg_gex.empty and not pos_gex.empty:
        idx_neg = neg_gex.index.max()
        val_neg = neg_gex.loc[idx_neg, 'total_gex']
        candidates_pos = pos_gex[pos_gex.index > idx_neg]
        if not candidates_pos.empty:
            idx_pos = candidates_pos.index.min()
            val_pos = candidates_pos.loc[idx_pos, 'total_gex']
            ratio = abs(val_neg) / (abs(val_neg) + val_pos)
            zero_gamma = idx_neg + (idx_pos - idx_neg) * ratio
        else:
            zero_gamma = subset['total_gex'].abs().idxmin()
    else:
        if not subset.empty:
            zero_gamma = subset['total_gex'].abs().idxmin()
    
    # ✅ NOUVEAU: Confidence score
    confidence = calculate_confidence(df, zero_gamma, spot)
    
    return df, call_wall, put_wall, zero_gamma, warnings, confidence

def calculate_confidence(df, zero_gamma, spot):
    """Score de confiance 0-100"""
    window = df[(df.index > zero_gamma * 0.98) & (df.index < zero_gamma * 1.02)]
    density_score = min(len(window) * 10, 40)
    
    distance_pct = abs(zero_gamma - spot) / spot
    distance_score = max(0, 30 - distance_pct * 100)
    
    neg_sum = abs(df[df['total_gex'] < 0]['total_gex'].sum())
    pos_sum = df[df['total_gex'] > 0]['total_gex'].sum()
    
    if neg_sum == 0 or pos_sum == 0:
        balance_score = 0
    else:
        ratio = min(neg_sum, pos_sum) / max(neg_sum, pos_sum)
        balance_score = ratio * 30
    
    return min(100, density_score + distance_score + balance_score)
import json
from pathlib import Path

# --- GESTION HISTORIQUE GEX ---
HISTORY_FILE = Path("gex_history.json")

def load_history():
    """Charge l'historique GEX"""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except:
        return []

def save_gex_snapshot(spot, call_wall, put_wall, zero_gamma, confidence):
    """Sauvegarde le snapshot actuel"""
    history = load_history()
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "spot": float(spot),
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
        "zero_gamma": float(zero_gamma),
        "confidence": float(confidence)
    }
    
    history.append(snapshot)
    
    # Garde les 168 dernières heures (7 jours si 1 calc/heure)
    history = history[-168:]
    
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    
    return snapshot

def detect_gex_shift():
    """Détecte les shifts significatifs"""
    history = load_history()
    
    if len(history) < 2:
        return None
    
    last = history[-1]
    previous = history[-2]
    
    # Calculs des shifts
    zg_shift = last['zero_gamma'] - previous['zero_gamma']
    zg_shift_pct = (zg_shift / previous['zero_gamma']) * 100
    
    cw_shift = last['call_wall'] - previous['call_wall']
    cw_shift_pct = (cw_shift / previous['call_wall']) * 100
    
    pw_shift = last['put_wall'] - previous['put_wall']
    pw_shift_pct = (pw_shift / previous['put_wall']) * 100
    
    # Temps écoulé
    time_diff = (datetime.fromisoformat(last['timestamp']) - 
                 datetime.fromisoformat(previous['timestamp']))
    
    # Détection des shifts critiques
    shifts = []
    
    if abs(zg_shift_pct) > 2.0:
        shifts.append({
            "type": "Zero Gamma",
            "old": previous['zero_gamma'],
            "new": last['zero_gamma'],
            "shift": zg_shift,
            "shift_pct": zg_shift_pct,
            "severity": "🚨 CRITIQUE" if abs(zg_shift_pct) > 5 else "⚠️ IMPORTANT"
        })
    
    if abs(cw_shift_pct) > 3.0:
        shifts.append({
            "type": "Call Wall",
            "old": previous['call_wall'],
            "new": last['call_wall'],
            "shift": cw_shift,
            "shift_pct": cw_shift_pct,
            "severity": "⚠️ MODÉRÉ"
        })
    
    if abs(pw_shift_pct) > 3.0:
        shifts.append({
            "type": "Put Wall",
            "old": previous['put_wall'],
            "new": last['put_wall'],
            "shift": pw_shift,
            "shift_pct": pw_shift_pct,
            "severity": "⚠️ MODÉRÉ"
        })
    
    if shifts:
        return {
            "shifts": shifts,
            "time_diff": time_diff,
            "timestamp": last['timestamp']
        }
    
    return None

def get_historical_chart_data():
    """Prépare les données pour le graphique historique"""
    history = load_history()
    
    if len(history) < 2:
        return None
    
    df_hist = pd.DataFrame(history)
    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
    
    # Garde les 48 dernières heures pour le chart
    df_hist = df_hist.tail(48)
    
    return df_hist

def analyze_gex_trend():
    """Analyse la tendance du GEX"""
    history = load_history()
    
    if len(history) < 5:
        return None
    
    # Prend les 10 derniers snapshots
    recent = history[-10:]
    
    zg_values = [s['zero_gamma'] for s in recent]
    
    # Calcul de la tendance (régression linéaire simple)
    x = np.arange(len(zg_values))
    z = np.polyfit(x, zg_values, 1)
    slope = z[0]
    
    # Interprétation
    avg_zg = np.mean(zg_values)
    slope_pct = (slope / avg_zg) * 100
    
    if slope_pct > 1.0:
        trend = "📈 HAUSSIER"
        interpretation = "Le Zero Gamma monte → Bias haussier structurel"
    elif slope_pct < -1.0:
        trend = "📉 BAISSIER"
        interpretation = "Le Zero Gamma baisse → Bias baissier structurel"
    else:
        trend = "➡️ STABLE"
        interpretation = "Le Zero Gamma est stable → Marché en équilibre"
    
    return {
        "trend": trend,
        "slope_pct": slope_pct,
        "interpretation": interpretation,
        "avg_zg": avg_zg
    }

# --- UTILITAIRES UI ---
def format_shift_alert(shift_data):
    """Formate joliment les alertes de shift"""
    alerts = []
    
    for shift in shift_data['shifts']:
        direction = "⬆️" if shift['shift'] > 0 else "⬇️"
        
        alert_text = f"""
**{shift['severity']} {shift['type']} Shift {direction}**

- Ancien : ${shift['old']:,.0f}  
- Nouveau : ${shift['new']:,.0f}  
- Variation : ${shift['shift']:,.0f} ({shift['shift_pct']:+.2f}%)  
- Temps : {shift_data['time_diff']}
        """
        alerts.append(alert_text.strip())
    
    return "\n\n".join(alerts)

def generate_trading_recommendation(spot, zero_gamma, call_wall, put_wall, confidence):
    """Génère une recommandation de trading"""
    
    distance_to_zg = spot - zero_gamma
    distance_pct = (distance_to_zg / zero_gamma) * 100
    
    distance_to_cw = call_wall - spot
    distance_to_pw = spot - put_wall
    
    # Logique de recommandation
    if abs(distance_pct) < 0.5:
        bias = "🟡 NEUTRE - Prix au Zero Gamma"
        action = "Attendre confirmation directionnelle. Zone de pivot."
        risk = "MOYEN"
        
    elif distance_pct < -2.0:  # Prix bien en dessous du ZG
        bias = "🟢 HAUSSIER - Prix sous Zero Gamma"
        action = f"Chercher LONG sur pullback. Target : ${zero_gamma:,.0f}"
        risk = "FAIBLE" if confidence > 70 else "MOYEN"
        
    elif distance_pct > 2.0:  # Prix bien au-dessus du ZG
        bias = "🔴 BAISSIER - Prix sur Zero Gamma"
        action = f"Chercher SHORT sur retest. Target : ${zero_gamma:,.0f}"
        risk = "FAIBLE" if confidence > 70 else "MOYEN"
        
    else:  # Entre -2% et +2%
        if distance_pct > 0:
            bias = "🟠 NEUTRE/BAISSIER - Légèrement sur ZG"
        else:
            bias = "🟠 NEUTRE/HAUSSIER - Légèrement sous ZG"
        action = "Zone d'indécision. Attendre."
        risk = "ÉLEVÉ"
    
    # Analyse des walls
    if distance_to_cw < distance_to_pw:
        wall_note = f"⚠️ Call Wall proche (${call_wall:,.0f}) = Résistance majeure"
    else:
        wall_note = f"⚠️ Put Wall proche (${put_wall:,.0f}) = Support majeur"
    
    return {
        "bias": bias,
        "action": action,
        "risk": risk,
        "wall_note": wall_note,
        "confidence_interpretation": (
            "✅ Haute fiabilité" if confidence > 70 else
            "⚠️ Fiabilité moyenne" if confidence > 50 else
            "❌ Faible fiabilité - Prudence"
        )
    }

# --- INTERFACE STREAMLIT ---
st.title("⏳ GEX Time Master Pro")
st.caption("Real-time Gamma Exposure Analysis with Smart Alerts")

# --- SECTION 1 : STATUS & HISTORIQUE ---
col_status1, col_status2 = st.columns([2, 1])

with col_status1:
    # Indicateur de dernière mise à jour
    history = load_history()
    if history:
        last_update = datetime.fromisoformat(history[-1]['timestamp'])
        time_since = datetime.now() - last_update
        minutes_ago = int(time_since.total_seconds() / 60)
        
        if minutes_ago < 60:
            st.info(f"📡 Dernière MAJ : Il y a {minutes_ago} min")
        else:
            hours_ago = minutes_ago // 60
            st.warning(f"⚠️ Dernière MAJ : Il y a {hours_ago}h {minutes_ago % 60}min")
    else:
        st.info("📡 Aucun historique - Premier calcul")

with col_status2:
    # Bouton refresh avec compteur
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- CHARGEMENT DONNÉES ---
if 'raw_data' not in st.session_state:
    with st.spinner("🔌 Connexion Deribit..."):
        s, d = get_deribit_data('BTC')
        if s and d:
            st.session_state['spot'] = s
            st.session_state['raw_data'] = d
            st.success("✅ Connecté")
        else:
            st.error("❌ Échec connexion")
            st.stop()

spot = st.session_state['spot']
data = st.session_state['raw_data']

# --- SECTION 2 : DÉTECTION AUTOMATIQUE DES SHIFTS ---
shift_detected = detect_gex_shift()

if shift_detected:
    st.markdown("---")
    st.markdown("### 🚨 ALERTE GEX SHIFT DÉTECTÉ")
    
    # Affichage des shifts dans des expandeurs
    for shift in shift_detected['shifts']:
        severity_color = "red" if "CRITIQUE" in shift['severity'] else "orange"
        
        with st.expander(f"{shift['severity']} {shift['type']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                "Ancien",
                f"${shift['old']:,.0f}",
                delta=None
            )
            
            col2.metric(
                "Nouveau", 
                f"${shift['new']:,.0f}",
                delta=f"{shift['shift_pct']:+.2f}%",
                delta_color="inverse" if shift['shift'] < 0 else "normal"
            )
            
            col3.metric(
                "Variation",
                f"${abs(shift['shift']):,.0f}",
                delta=f"Il y a {shift_detected['time_diff']}"
            )
            
            # Action recommandée
            if shift['type'] == "Zero Gamma":
                if shift['shift'] > 0:
                    st.warning("⬆️ **GEX monte** → Prix sera attiré VERS LE HAUT")
                else:
                    st.warning("⬇️ **GEX baisse** → Prix sera attiré VERS LE BAS")
                
                st.info("📋 **Action** : Mettez à jour votre TradingView MAINTENANT !")
    
    st.markdown("---")

# --- SECTION 3 : CALENDRIER EXPIRATIONS ---
st.markdown("### 📅 Calendrier des Expirations Majeures")

sorted_days, exp_details = analyze_upcoming_expirations(data)

if sorted_days:
    # Affiche les 4 prochaines
    cols = st.columns(min(4, len(sorted_days)))
    for i in range(min(4, len(sorted_days))):
        days = sorted_days[i]
        info = exp_details[days]
        
        # Couleur selon urgence
        if days < 7:
            color = "🔴"
        elif days < 30:
            color = "🟡"
        else:
            color = "🟢"
        
        cols[i].metric(
            label=f"{color} {info['type']}", 
            value=f"{days}j",
            delta=info['date'],
            delta_color="off"
        )

st.divider()

# --- SECTION 4 : PARAMÈTRES ---
st.markdown("### ⚙️ Configuration de l'Analyse")

col1, col2, col3 = st.columns(3)

with col1:
    dte_limit = st.slider(
        "📏 Horizon (Jours)", 
        min_value=1, 
        max_value=365, 
        value=65,
        help="Durée maximale des contrats à inclure"
    )

with col2:
    only_fridays = st.checkbox(
        "📆 Vendredis uniquement", 
        value=True,
        help="Filtre pour ne garder que les expirations du vendredi"
    )

with col3:
    use_weighting = st.checkbox(
        "⚖️ Pondération intelligente", 
        value=True,
        help="Donne plus de poids aux quarterly/monthly"
    )

# Poids avancés (cachés dans un expander)
with st.expander("🎛️ Réglages avancés (Experts uniquement)"):
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        w_quart = st.number_input("Poids Quarterly", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
    with col_w2:
        w_month = st.number_input("Poids Monthly", value=2.0, min_value=1.0, max_value=10.0, step=0.5)
    with col_w3:
        w_week = st.number_input("Poids Weekly", value=1.0, min_value=0.5, max_value=5.0, step=0.5)

st.divider()

# --- SECTION 5 : ANALYSE TENDANCE HISTORIQUE ---
trend_data = analyze_gex_trend()

if trend_data:
    st.markdown("### 📊 Analyse de Tendance GEX (10 derniers calculs)")
    
    col_t1, col_t2 = st.columns([1, 2])
    
    with col_t1:
        st.metric(
            "Tendance Zero Gamma",
            trend_data['trend'],
            delta=f"{trend_data['slope_pct']:.2f}% par calcul"
        )
    
    with col_t2:
        st.info(f"💡 **Interprétation** : {trend_data['interpretation']}")

# --- SECTION 6 : BOUTON CALCUL PRINCIPAL ---
if st.button("🚀 CALCULER LE GEX", type="primary", use_container_width=True):
    
    with st.spinner("🧮 Calcul en cours..."):
        df, cw, pw, zg, warns, conf = process_gex(
            spot, data, dte_limit, only_fridays, 
            use_weighting, w_quart, w_month, w_week
        )
    
    # Sauvegarde dans l'historique
    save_gex_snapshot(spot, cw, pw, zg, conf)
    
    # Affichage warnings
    if warns:
        for w in warns:
            st.warning(w)
    
    if not df.empty:
        st.markdown("---")
        st.markdown("## 📈 Résultats de l'Analyse")
        
        # --- METRICS PRINCIPALES ---
        st.markdown(f"### Prix Spot BTC : **${spot:,.2f}**")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "🔴 Call Wall",
                f"${cw:,.0f}",
                delta=f"+{((cw - spot) / spot * 100):.1f}%" if cw > spot else f"{((cw - spot) / spot * 100):.1f}%"
            )
        
        with col_m2:
            st.metric(
                "🟢 Put Wall",
                f"${pw:,.0f}",
                delta=f"{((pw - spot) / spot * 100):.1f}%"
            )
        
        with col_m3:
            zg_delta = ((zg - spot) / spot * 100)
            st.metric(
                "⚖️ Zero Gamma",
                f"${zg:,.0f}",
                delta=f"{zg_delta:+.2f}%"
            )
        
        with col_m4:
            # Couleur selon confidence
            conf_color = "🟢" if conf > 70 else "🟡" if conf > 50 else "🔴"
            st.metric(
                f"{conf_color} Confiance",
                f"{conf:.0f}%",
                delta="Haute" if conf > 70 else "Moyenne" if conf > 50 else "Faible"
            )
        
        st.divider()
        
        # --- RECOMMANDATION DE TRADING ---
        st.markdown("### 🎯 Recommandation de Trading")
        
        reco = generate_trading_recommendation(spot, zg, cw, pw, conf)
        
        col_r1, col_r2 = st.columns([2, 1])
        
        with col_r1:
            st.markdown(f"""
**{reco['bias']}**

📍 **Action suggérée** : {reco['action']}

{reco['wall_note']}

🎲 **Niveau de risque** : {reco['risk']}
            """)
        
        with col_r2:
            st.info(f"**Fiabilité du Zero Gamma**\n\n{reco['confidence_interpretation']}")
        
        st.divider()
        
        # --- GRAPHIQUE GEX ---
        st.markdown("### 📊 Distribution du Gamma Exposure")
        
        # Filtre pour le graphique (±30% du spot)
        df_chart = df[(df.index > spot * 0.7) & (df.index < spot * 1.3)].reset_index()
        
        # Graphique principal
        base = alt.Chart(df_chart).encode(
            x=alt.X('Strike:Q', 
                   axis=alt.Axis(format='$,.0f', title='Strike Price'),
                   scale=alt.Scale(domain=[spot * 0.85, spot * 1.15]))
        )
        
        # Barres GEX
        bars = base.mark_bar(size=3).encode(
            y=alt.Y('total_gex:Q', title='GEX (Millions)'),
            color=alt.condition(
                alt.datum.total_gex > 0,
                alt.value('#00C853'),  # Vert pour positif
                alt.value('#D50000')   # Rouge pour négatif
            ),
            tooltip=[
                alt.Tooltip('Strike:Q', format='$,.0f', title='Strike'),
                alt.Tooltip('total_gex:Q', format=',.2f', title='GEX')
            ]
        )
        
        # Ligne du spot
        spot_line = alt.Chart(pd.DataFrame({'spot': [spot]})).mark_rule(
            color='yellow',
            strokeWidth=2,
            strokeDash=[5, 5]
        ).encode(
            x='spot:Q'
        )
        
        # Ligne Zero Gamma
        zg_line = alt.Chart(pd.DataFrame({'zg': [zg]})).mark_rule(
            color='white',
            strokeWidth=2
        ).encode(
            x='zg:Q'
        )
        
        # Texte annotations
        annotations = alt.Chart(pd.DataFrame({
            'x': [spot, zg],
            'y': [df_chart['total_gex'].max() * 0.9] * 2,
            'text': ['Spot', 'Zero Gamma']
        })).mark_text(
            align='center',
            baseline='middle',
            fontSize=12,
            fontWeight='bold',
            color='white'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        
        chart = (bars + spot_line + zg_line + annotations).properties(
            height=400
        ).configure_view(
            strokeWidth=0
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        st.divider()
        
        # --- GRAPHIQUE HISTORIQUE (si dispo) ---
        hist_data = get_historical_chart_data()
        
        if hist_data is not None and len(hist_data) > 1:
            st.markdown("### 📈 Évolution du Zero Gamma (48h)")
            
            # Graphique ligne temporel
            hist_chart = alt.Chart(hist_data).mark_line(
                point=True,
                color='#2962FF',
                strokeWidth=2
            ).encode(
                x=alt.X('timestamp:T', title='Temps', axis=alt.Axis(format='%d/%m %H:%M')),
                y=alt.Y('zero_gamma:Q', title='Zero Gamma ($)', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('timestamp:T', format='%d %b %H:%M', title='Date'),
                    alt.Tooltip('zero_gamma:Q', format='$,.0f', title='Zero Gamma'),
                    alt.Tooltip('confidence:Q', format='.0f', title='Confiance (%)')
                ]
            ).properties(height=250).interactive()
            
            # Ligne du ZG actuel
            current_zg_line = alt.Chart(pd.DataFrame({'zg': [zg]})).mark_rule(
                color='yellow',
                strokeDash=[5, 5]
            ).encode(y='zg:Q')
            
            st.altair_chart(hist_chart + current_zg_line, use_container_width=True)
        
        st.divider()
        
        # --- CODE TRADINGVIEW ---
        st.markdown("### 📋 Code pour TradingView (Pine Script)")
        
        # Génération du code avec timestamp
        tv_code = f"""// ═══════════════════════════════════════════════════════════
// GEX Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
// Confiance: {conf:.0f}% | Spot: ${spot:,.2f}
// ═══════════════════════════════════════════════════════════

float call_wall = {cw}
float put_wall = {pw}
float zero_gamma = {zg}

// Distance au Zero Gamma: {((zg - spot) / spot * 100):+.2f}%
// Bias: {"HAUSSIER" if spot < zg else "BAISSIER" if spot > zg else "NEUTRE"}"""
        
        col_code1, col_code2 = st.columns([4, 1])
        
        with col_code1:
            st.code(tv_code, language='pine')
        
        with col_code2:
            # Bouton copie (nécessite pyperclip, sinon fallback)
            if st.button("📋 Copier", key="copy_btn", use_container_width=True):
                try:
                    import pyperclip
                    pyperclip.copy(tv_code)
                    st.success("✅ Copié !")
                except ImportError:
                    st.warning("⚠️ Pour copier automatiquement : `pip install pyperclip`")
                    st.info("💡 Sélectionnez et copiez manuellement")
        
        # Export CSV (optionnel)
        with st.expander("💾 Exporter les données"):
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f'gex_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )
            
            # Export historique JSON
            hist_json = json.dumps(load_history(), indent=2)
            st.download_button(
                label="📥 Télécharger Historique JSON",
                data=hist_json,
                file_name=f'gex_history_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
                mime='application/json',
            )
    
    else:
        st.error("❌ Aucune donnée à afficher. Vérifiez vos filtres.")

else:
    # Message si pas encore calculé
    st.info("👆 Configurez les paramètres et cliquez sur **CALCULER LE GEX**")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>⏳ GEX Time Master Pro v2.0 | Data: Deribit API</p>
    <p>⚠️ Trading comporte des risques. Ceci n'est pas un conseil financier.</p>
</div>
""", unsafe_allow_html=True)
