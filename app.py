import sys
import os
import streamlit as st
import pandas as pd
import duckdb
import json
import re

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.config import ARTIFACTS_DIR, DB_PATH
from scripts.case_management import (
    CaseManagementError,
    CaseStatus,
    CaseStore,
    ReviewerRole,
    event_to_dict,
)
from scripts.inference_engine import load_models, run_inference
from scripts.pii import mask_pan
from scripts.sar_agent import generate_sar_narrative, save_sar_report

# Page config
st.set_page_config(
    page_title="AegisAgent Investigation Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode & Premium Accents)
st.markdown("""
<style>
    /* Main container background and color */
    .stApp {
        background-color: #080b11;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Custom App Title */
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #d2a8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Custom metric card */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 24px;
        flex: 1;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        backdrop-filter: blur(8px);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(88, 166, 255, 0.4);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.1);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    /* Risk gauge styles */
    .gauge-breached {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.1) 0%, rgba(248, 81, 73, 0.03) 100%);
        border: 1.5px solid rgba(248, 81, 73, 0.4);
        border-radius: 12px;
        padding: 22px;
        color: #ff7b72;
        box-shadow: 0 8px 32px rgba(248, 81, 73, 0.08);
        backdrop-filter: blur(8px);
    }
    .gauge-safe {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.08) 0%, rgba(56, 139, 253, 0.02) 100%);
        border: 1.5px solid rgba(56, 139, 253, 0.3);
        border-radius: 12px;
        padding: 22px;
        color: #58a6ff;
        box-shadow: 0 8px 32px rgba(56, 139, 253, 0.05);
        backdrop-filter: blur(8px);
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #090d14;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Custom divider */
    .custom-hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(48, 54, 61, 0.2), #58a6ff, rgba(48, 54, 61, 0.2));
        margin: 30px 0;
    }
    
    /* Compliance Document Layout */
    .report-card-container {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.9) 100%);
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .report-card-container:hover {
        transform: translateY(-3px);
    }
    
    /* Section-specific hovers */
    .report-card-who:hover { border-color: #58a6ff !important; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(88, 166, 255, 0.15) !important; }
    .report-card-what:hover { border-color: #79c0ff !important; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(121, 192, 255, 0.15) !important; }
    .report-card-when:hover { border-color: #56d364 !important; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(86, 211, 100, 0.15) !important; }
    .report-card-where:hover { border-color: #f8e3a1 !important; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(248, 227, 161, 0.15) !important; }
    .report-card-why:hover { border-color: #ff7b72 !important; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(255, 123, 114, 0.15) !important; }
    .report-card-how:hover { border-color: #d2a8ff !important; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(210, 168, 255, 0.15) !important; }
    .report-card-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
    }
    .report-card-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: rgba(var(--accent-rgb, 88, 166, 255), 0.1);
        color: var(--accent-color, #58a6ff);
        border: 1px solid rgba(var(--accent-rgb, 88, 166, 255), 0.2);
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .report-card-body {
        color: #e6edf3;
        font-size: 1rem;
        line-height: 1.7;
    }
    .report-card-body strong {
        color: #ffffff;
        font-weight: 600;
    }
    .report-card-body p {
        margin-bottom: 14px;
    }
    .report-card-body p:last-child {
        margin-bottom: 0;
    }
    .report-card-body ol, .report-card-body ul {
        margin-top: 8px;
        margin-bottom: 14px;
        padding-left: 24px;
    }
    .report-card-body li {
        margin-bottom: 8px;
        line-height: 1.6;
    }
    .report-card-body li:last-child {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load model version telemetry
def load_telemetry():
    artifacts_dir = str(ARTIFACTS_DIR)
    latest_file = os.path.join(artifacts_dir, "latest_version.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            version_str = f.read().strip()
        metrics_file = os.path.join(artifacts_dir, version_str, "training_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f), version_str
    return None, "Unknown"

# Cache resource for models
@st.cache_resource
def load_production_models():
    return load_models(ARTIFACTS_DIR)

# Load database transactions
@st.cache_data
def get_db_transactions():
    db_path = str(DB_PATH)
    if not os.path.exists(db_path):
        return []
    
    con = duckdb.connect(db_path)
    # Join features with raw data fields from stg_transactions_test to get names/merchant names/etc.
    query = """
        SELECT 
            f.*, 
            t.first, 
            t.last, 
            t.gender, 
            t.street, 
            t.city, 
            t.state, 
            t.zip, 
            t.lat, 
            t.long, 
            t.merchant, 
            t.category, 
            t.merch_lat, 
            t.merch_long, 
            t.job, 
            t.dob
        FROM fct_fraud_features f
        LEFT JOIN stg_transactions_test t ON f.trans_num = t.trans_num
        WHERE f.evaluation_role = 'development_holdout'
        LIMIT 200
    """
    df = con.execute(query).df()
    con.close()
    return df.to_dict(orient='records')

# A synthetic high-risk transaction used to exercise the STR path on demand.
# Only the *input* is synthetic -- it carries a complete feature vector and is
# scored by the real ensemble like every other row in the dropdown. Nothing in
# this dashboard displays a model score that did not come out of a model.
FRAUD_DEMO_TXN = {
    'trans_num': 'TXN_88888',
    'trans_date_trans_time': pd.Timestamp('2026-06-10 02:45:00'),
    'cc_num': 9876543210987654,
    'first': 'Alice',
    'last': 'Smith',
    'gender': 'F',
    'job': 'Data Engineer',
    'street': '123 Main Street',
    'city': 'Boston',
    'state': 'MA',
    'zip': '02108',
    'lat': 42.3601,
    'long': -71.0589,
    'merchant': 'fraud_Luxury_Jewelry_Store',
    'category': 'shopping_pos',
    'merch_lat': 51.5074,
    'merch_long': -0.1278,
    'amt': 7500.00,
    'distance_km': 5262.11,
    'night': 1,
    'card_mean_amt': 50.00,
    'amt_z_card': 149.00,
    'txns_24h': 15,
    'txns_7d': 25,
    'category_risk': 0.015,
    'log_amt': 8.9227,
    'hour': 2,
    'day_of_week': 2,
    'hour_sin': -0.2588,
    'hour_cos': 0.9659,
    'is_online': 1,
    'state_risk': 0.0034,
    'merchant_risk': 0.0153,
    'card_txn_cnt': 480,
    'card_std_amt': 154.27,
    'amt_x_catRisk': 112.5,
    'dist_x_online': 5262.11,
    'amt_x_night': 7500.00,
    'is_fraud': 1
}

# Main Application Title
st.markdown('<h1 class="app-title">🛡️ AegisAgent: Fraud Investigation Support Demo</h1>', unsafe_allow_html=True)
st.markdown(
    "Explore historical synthetic card transactions, inspect ensemble alert scores, "
    "and generate a 5W+H investigation narrative draft. This demo does not make a "
    "reasonable-grounds-to-suspect (RGS) determination or submit a report to FINTRAC."
)

# Sidebar Controls
st.sidebar.header("🔍 Transaction Selection")

st.sidebar.subheader("👤 Human reviewer")
reviewer_id = st.sidebar.text_input(
    "Reviewer identifier",
    help="Use a stable workforce identifier. This demo does not authenticate it.",
).strip()
reviewer_role = st.sidebar.selectbox(
    "Workflow role",
    options=[role.value for role in ReviewerRole],
    format_func=lambda value: value.replace("_", " ").title(),
    help=(
        "Role selection is self-attested in this local demo. Production authorization "
        "and segregation-of-duties controls remain out of scope."
    ),
)

# Load data and prepare dropdown options
txns = get_db_transactions()
# Add demo transaction at index 0
all_options = [FRAUD_DEMO_TXN] + txns

options_labels = []
for t in all_options:
    lbl = f"{t['trans_num']} - ${t['amt']:.2f} ({t['category']})"
    if t['trans_num'] == 'TXN_88888':
        lbl += " ⚠️ FRAUD DEMO"
    elif t['is_fraud'] == 1:
        lbl += " [FRAUD]"
    options_labels.append(lbl)

selected_idx = st.sidebar.selectbox(
    "Choose Transaction",
    range(len(all_options)),
    format_func=lambda i: options_labels[i]
)

selected_txn = all_options[selected_idx]
st.sidebar.caption(f"Card: {mask_pan(selected_txn.get('cc_num'))}")

# Sidebar: System Metadata
st.sidebar.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.sidebar.subheader("⚙️ System Metadata")

telemetry, model_version = load_telemetry()
st.sidebar.write("**Model Directory:** `models_artifacts/`")
st.sidebar.write(f"**Active Version:** `{model_version}`")

if telemetry:
    st.sidebar.write(f"**Trained At:** `{pd.Timestamp(telemetry.get('trained_at')).strftime('%Y-%m-%d %H:%M:%S')}`")
    st.sidebar.write(f"**Decision Threshold:** `{telemetry.get('optimal_threshold', 0.0):.4f}`")
    st.sidebar.caption(f"Selected on: {telemetry.get('threshold_selected_on', 'n/a')}")

    st.sidebar.markdown("**Development-holdout performance**")
    st.sidebar.write(f"- PR AUC: `{telemetry.get('development_pr_auc_meta', telemetry.get('test_pr_auc_meta', 0.0)):.4f}`")
    st.sidebar.write(f"- ROC AUC: `{telemetry.get('development_auc_meta', telemetry.get('test_auc_meta', 0.0)):.4f}`")
    st.sidebar.write(f"- Precision: `{telemetry.get('development_precision', telemetry.get('test_precision', 0.0)):.2%}`")
    st.sidebar.write(f"- Recall: `{telemetry.get('development_recall', telemetry.get('test_recall', 0.0)):.2%}`")
    st.sidebar.caption(
        "PR AUC is the headline metric here: at ~0.4% fraud prevalence, ROC AUC "
        "flatters every model."
    )

    development_alerts = telemetry.get('development_alerts', telemetry.get('test_alerts'))
    if development_alerts:
        st.sidebar.markdown("**Operational load**")
        st.sidebar.write(f"- Alerts: `{development_alerts:,}`")
        st.sidebar.write(f"- Per day: `{telemetry.get('development_alerts_per_day', telemetry.get('test_alerts_per_day', 0))}`")

    with st.sidebar.expander("Base Model Performance (Development ROC AUC)"):
        st.sidebar.write(f"- Model 2 (Geo): `{telemetry.get('development_auc_m2', telemetry.get('test_auc_m2', 0.0)):.4f}`")
        st.sidebar.write(f"- Model 3 (Cat): `{telemetry.get('development_auc_m3', telemetry.get('test_auc_m3', 0.0)):.4f}`")
        st.sidebar.write(f"- Model 4 (Vel): `{telemetry.get('development_auc_m4', telemetry.get('test_auc_m4', 0.0)):.4f}`")

# 1. Metrics Layer (Displaying the 4 key values with clean icons)
st.subheader("📊 Transaction Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💵 Transaction Amount</div>
        <div class="metric-value">${selected_txn['amt']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    clean_cat = selected_txn['category'].replace('_', ' ').title() if selected_txn['category'] else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏷️ Merchant Category</div>
        <div class="metric-value">{clean_cat}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🗺️ Geographic Distance</div>
        <div class="metric-value">{selected_txn['distance_km']:.2f} km</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    night_str = "Yes 🌙" if selected_txn['night'] == 1 else "No ☀️"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🕒 Night Indicator</div>
        <div class="metric-value">{night_str}</div>
    </div>
    """, unsafe_allow_html=True)

# 2. Inference Telemetry Section
st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.subheader("🧠 Stacking Ensemble Telemetry")

# Load models and threshold
model_2, model_3, model_4, scaler_4, meta_model, threshold = load_production_models()

# Every transaction, including the synthetic demo row, is scored by the live
# ensemble. There is deliberately no override branch here.
df_row = pd.DataFrame([selected_txn])
try:
    meta_p, p_m2, p_m3, p_m4, triggered_alert = run_inference(
        df_row, model_2, model_3, model_4, scaler_4, meta_model, threshold
    )
    meta_score = float(meta_p[0])
    p_m2_val = float(p_m2[0])
    p_m3_val = float(p_m3[0])
    p_m4_val = float(p_m4[0])
    triggered = bool(triggered_alert[0])
except Exception as e:
    st.error(f"Inference Engine Error: {e}")
    st.stop()

# Layout: Base model progress bars (left) + Risk Gauge (right)
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.write("##### Base Model Probability Outputs")
    
    # Model 2 Progress Bar
    st.markdown("**Model 2 (Geographic Focus)** - Geographic anomaly index")
    st.progress(p_m2_val)
    st.markdown(f"<p style='text-align: right; margin-top:-15px; color:#8b949e; font-size:0.9rem;'>Score: <b>{p_m2_val:.4f}</b></p>", unsafe_allow_html=True)
    
    # Model 3 Progress Bar
    st.markdown("**Model 3 (Category Focus)** - Category spend anomaly index")
    st.progress(p_m3_val)
    st.markdown(f"<p style='text-align: right; margin-top:-15px; color:#8b949e; font-size:0.9rem;'>Score: <b>{p_m3_val:.4f}</b></p>", unsafe_allow_html=True)
    
    # Model 4 Progress Bar
    st.markdown("**Model 4 (Velocity Focus)** - Card transaction rate and velocity index")
    st.progress(p_m4_val)
    st.markdown(f"<p style='text-align: right; margin-top:-15px; color:#8b949e; font-size:0.9rem;'>Score: <b>{p_m4_val:.4f}</b></p>", unsafe_allow_html=True)

with col_right:
    st.write("##### Staked Classifier Assessment")
    
    # Risk gauge visualization
    risk_color = "#ff7b72" if triggered else "#58a6ff"
    gradient = "linear-gradient(to right, #58a6ff, #f0883e, #ff7b72)" if triggered else "linear-gradient(to right, #58a6ff, #388bfd)"
    status_text = "⚠️ HIGH RISK - BREACH DETECTED" if triggered else "🟢 SECURE - INSIDE PARAMETERS"
    gauge_class = "gauge-breached" if triggered else "gauge-safe"
    
    st.markdown(f"""
    <div class="{gauge_class}">
        <h4 style="margin-top:0; color:{risk_color} !important; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">{status_text}</h4>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 10px; margin-bottom: 5px;">
            <span style="font-size: 2.2rem; font-weight: 800; color: #f0f6fc;">{meta_score * 100:.2f}%</span>
            <span style="font-size: 0.9rem; color: #8b949e;">Meta-Score</span>
        </div>
        <div style="background-color: #21262d; border-radius: 4px; height: 12px; width: 100%; position: relative; border: 1px solid #30363d; overflow: hidden; margin-top: 10px;">
            <div style="background: {gradient}; height: 100%; width: {meta_score * 100:.1f}%;"></div>
            <div style="position: absolute; left: {threshold * 100}%; top: 0; bottom: 0; width: 2.5px; background-color: #f0f6fc; box-shadow: 0 0 6px #ffffff;" title="Threshold: {threshold:.4f}"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #8b949e; margin-top: 6px;">
            <span>0% (Safe)</span>
            <span style="color: {risk_color}; font-weight: 600;">Threshold Limit: {threshold:.4f} ({threshold * 100:.1f}%)</span>
            <span>100% (Threat)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 3. Investigation Narrative Preview Section
st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.subheader("📋 Investigation narrative drafting assistant")

def markdown_to_html(text):
    # Convert bold **text** to <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert italic *text* to <em>text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    lines = text.split("\n")
    html_lines = []
    in_ordered_list = False
    in_unordered_list = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Check if numbered list item (e.g. 1. Velocity Anomaly)
        ol_match = re.match(r'^(\d+)\.\s+(.*)$', line_stripped)
        # Check if bullet list item (e.g. - or * Velocity Anomaly)
        ul_match = re.match(r'^[\*\-\+]\s+(.*)$', line_stripped)
        
        if ol_match:
            if in_unordered_list:
                html_lines.append('</ul>')
                in_unordered_list = False
            if not in_ordered_list:
                html_lines.append('<ol style="margin-top: 8px; margin-bottom: 8px; padding-left: 20px;">')
                in_ordered_list = True
            html_lines.append(f'<li style="margin-bottom: 8px; line-height: 1.6;">{ol_match.group(2)}</li>')
        elif ul_match:
            if in_ordered_list:
                html_lines.append('</ol>')
                in_ordered_list = False
            if not in_unordered_list:
                html_lines.append('<ul style="margin-top: 8px; margin-bottom: 8px; padding-left: 20px; list-style-type: disc;">')
                in_unordered_list = True
            html_lines.append(f'<li style="margin-bottom: 8px; line-height: 1.6;">{ul_match.group(2)}</li>')
        else:
            if in_ordered_list:
                html_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                html_lines.append('</ul>')
                in_unordered_list = False
                
            # Check for headers
            if line_stripped.startswith("###"):
                html_lines.append(f'<h5 style="margin-top: 16px; margin-bottom: 8px; color: #58a6ff; font-weight: 600;">{line_stripped[3:].strip()}</h5>')
            elif line_stripped.startswith("##"):
                html_lines.append(f'<h4 style="margin-top: 20px; margin-bottom: 10px; color: #58a6ff; font-weight: 600;">{line_stripped[2:].strip()}</h4>')
            elif line_stripped.startswith("#"):
                html_lines.append(f'<h3 style="margin-top: 24px; margin-bottom: 12px; color: #58a6ff; font-weight: 600;">{line_stripped[1:].strip()}</h3>')
            else:
                html_lines.append(f'<p style="margin-bottom: 12px; line-height: 1.6;">{line_stripped}</p>')
                
    if in_ordered_list:
        html_lines.append('</ol>')
    if in_unordered_list:
        html_lines.append('</ul>')
        
    return "\n".join(html_lines)

# Helper function to parse LLM response into structured card elements
def render_parsed_compliance_report(narrative):
    if not narrative:
        return
        
    # Standard clean up
    lines = narrative.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("# suspicious transaction report") or \
           stripped.lower().startswith("## fintrac-compliant") or \
           stripped == "---":
            continue
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # 5W+H sections and mapping configs
    sections = {
        "WHO": {"icon": "👤", "title": "Who (Subject Profile)", "color": "#58a6ff", "rgb": "88, 166, 255"},
        "WHAT": {"icon": "💳", "title": "What (Transaction Facts)", "color": "#79c0ff", "rgb": "121, 192, 255"},
        "WHEN": {"icon": "📅", "title": "When (Temporal Context)", "color": "#56d364", "rgb": "86, 211, 100"},
        "WHERE": {"icon": "📍", "title": "Where (Geographic Mapping)", "color": "#f8e3a1", "rgb": "248, 227, 161"},
        "WHY": {"icon": "🔍", "title": "Why (Signals for investigator assessment)", "color": "#ff7b72", "rgb": "255, 123, 114"},
        "HOW": {"icon": "⚙️", "title": "How (Ensemble Alert consensus)", "color": "#d2a8ff", "rgb": "210, 168, 255"}
    }
    
    # regex split to partition by uppercase section titles (supporting headers like ## WHO, WHO, **WHO**)
    pattern = r"(?:^|\n)(?:\d+\.\s*)?[\s#\*]*(WHO|WHAT|WHEN|WHERE|WHY|HOW)[\s#\*:\-]*(?:\n|$)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    if len(parts) <= 1:
        # Fallback if partition fails
        st.markdown(f"""
        <div class="report-card-container" style="border-left: 5px solid #ff7b72; border-color: #ff7b72;">
            <div class="report-card-header">
                <span class="report-card-badge" style="background-color: rgba(255, 123, 114, 0.1); color: #ff7b72; border: 1px solid rgba(255, 123, 114, 0.2);">📋 Suspicious Activity Narrative Report</span>
            </div>
            <div class="report-card-body" style="font-family: monospace; white-space: pre-wrap;">{narrative}</div>
        </div>
        """, unsafe_allow_html=True)
        return
        
    # Render intro paragraph if any exists
    intro = parts[0].strip()
    if intro:
        st.markdown(f"<div style='margin-bottom: 20px; color: #8b949e; font-style: italic; line-height: 1.5;'>{intro}</div>", unsafe_allow_html=True)
        
    # Render sections inside cards
    for i in range(1, len(parts), 2):
        sec_name = parts[i].upper().strip()
        sec_content = parts[i+1].strip() if i+1 < len(parts) else ""
        
        if sec_name in sections:
            info = sections[sec_name]
            # Clean list formatting formatting issues (e.g. newline between numbers and texts)
            # Match only numbers followed by period and newline at the start of a line or right after a newline
            sec_content = re.sub(r"(^|\n)(\d+)\.\s*\n\s*", r"\1\2. ", sec_content)
            sec_content = re.sub(r"(^|\n)([\*\-\+])\s*\n\s*", r"\1\2 ", sec_content)
            
            html_content = markdown_to_html(sec_content)
            
            # Draw section wrapper card in a single st.markdown call
            sec_class = f"report-card-{sec_name.lower()}"
            st.markdown(f"""
            <div class="report-card-container {sec_class}" style="border-left: 5px solid {info['color']};">
                <div class="report-card-header">
                    <span class="report-card-badge" style="background-color: rgba({info['rgb']}, 0.1); color: {info['color']}; border: 1px solid rgba({info['rgb']}, 0.2);">{info['icon']}&nbsp;&nbsp;{info['title']}</span>
                </div>
                <div class="report-card-body">
                    {html_content}
                </div>
            </div>
            """, unsafe_allow_html=True)

if triggered:
    st.warning(
        "The transaction exceeded the model's alert threshold and should be reviewed. "
        "An alert is not an RGS determination and does not by itself require an STR. "
        "A reporting entity's authorized investigator must assess the facts, context, "
        "and applicable ML/TF indicators."
    )

    st.markdown("#### Human review case")
    case_store = CaseStore()
    case_record = case_store.find_by_transaction(selected_txn["trans_num"])

    if case_record is None:
        st.caption(
            "No case exists for this alert. Creating one records the score, threshold, "
            "model version, actor, and timestamp; it does not make an RGS determination."
        )
        if st.button(
            "Create Investigation Case",
            disabled=len(reviewer_id) < 2,
            use_container_width=True,
        ):
            try:
                case_store.create_alert_case(
                    trans_num=selected_txn["trans_num"],
                    model_score=meta_score,
                    threshold=threshold,
                    model_version=model_version,
                    actor=reviewer_id,
                    actor_role=reviewer_role,
                    metadata={"source": "streamlit_dashboard"},
                )
                st.rerun()
            except CaseManagementError as exc:
                st.error(f"Case creation rejected: {exc}")
        if len(reviewer_id) < 2:
            st.info("Enter a reviewer identifier in the sidebar to create a case.")
        st.stop()

    status_labels = {
        CaseStatus.ALERT_OPEN.value: "Alert open — awaiting assignment",
        CaseStatus.UNDER_REVIEW.value: "Under human review",
        CaseStatus.RGS_NOT_REACHED.value: "Closed — RGS not reached",
        CaseStatus.RGS_REACHED.value: "Closed — RGS reached; approved reporting workflow required",
    }
    st.write(f"**Case:** `{case_record.case_id}`")
    st.write(f"**Status:** {status_labels[case_record.status]}")
    st.write(f"**Assigned to:** `{case_record.assigned_to or 'Unassigned'}`")
    st.caption(f"Case version {case_record.version} · updated {case_record.updated_at}")

    with st.expander("Case event history"):
        history_rows = [event_to_dict(event) for event in case_store.history(case_record.case_id)]
        for row in history_rows:
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
        st.dataframe(history_rows, use_container_width=True, hide_index=True)

    evidence_rows = case_store.list_evidence(case_record.case_id)
    with st.expander(f"Evidence inventory ({len(evidence_rows)})"):
        if evidence_rows:
            st.dataframe(
                [
                    {
                        "evidence_id": item.evidence_id,
                        "type": item.evidence_type,
                        "sha256": item.sha256,
                        "bytes": item.byte_size,
                        "archive_verified": bool(
                            item.archive_receipt and item.archive_receipt.get("verified")
                        ),
                        "created_at": item.created_at,
                    }
                    for item in evidence_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No evidence artifacts are linked to this case.")
        if st.button("Verify Case & Evidence Integrity"):
            integrity = case_store.verify_integrity(case_record.case_id)
            if integrity["ok"]:
                st.success(
                    f"Verified {integrity['events_verified']} events and "
                    f"{integrity['evidence_verified']} evidence artifacts. "
                    f"Chain head: `{integrity['chain_head']}`"
                )
            else:
                st.error("Integrity verification failed: " + "; ".join(integrity["issues"]))

    if case_record.status == CaseStatus.ALERT_OPEN.value:
        review_rationale = st.text_area(
            "Assignment rationale",
            placeholder="Document why this alert is being assigned for investigation (20+ characters).",
        )
        if st.button(
            "Assign to Me & Start Review",
            disabled=len(reviewer_id) < 2,
            use_container_width=True,
        ):
            try:
                case_store.start_review(
                    case_record.case_id,
                    actor=reviewer_id,
                    actor_role=reviewer_role,
                    rationale=review_rationale,
                    expected_version=case_record.version,
                )
                st.rerun()
            except CaseManagementError as exc:
                st.error(f"Review transition rejected: {exc}")
        st.stop()

    if case_record.status == CaseStatus.UNDER_REVIEW.value:
        st.info(
            "Narrative drafting remains optional. The disposition below must be based on "
            "the investigator's assessment of the available facts and institutional policy."
        )
        with st.expander("Record authorized RGS disposition"):
            decision = st.radio(
                "Disposition",
                options=("RGS not reached", "RGS reached"),
                horizontal=True,
            )
            decision_rationale = st.text_area(
                "Decision rationale",
                placeholder=(
                    "Document the facts and indicators supporting the human decision "
                    "(20+ characters)."
                ),
            )
            st.caption(
                "Only the authorized_rgs_reviewer role can save a disposition. Selecting "
                "that role here is a demo assertion, not production authentication."
            )
            if st.button("Save RGS Disposition", use_container_width=True):
                try:
                    case_store.record_rgs_decision(
                        case_record.case_id,
                        reached=decision == "RGS reached",
                        actor=reviewer_id,
                        actor_role=reviewer_role,
                        rationale=decision_rationale,
                        expected_version=case_record.version,
                    )
                    st.rerun()
                except CaseManagementError as exc:
                    st.error(f"RGS disposition rejected: {exc}")
    else:
        if case_record.status == CaseStatus.RGS_REACHED.value:
            st.warning(
                "RGS was recorded by the human reviewer. This project does not prepare, "
                "approve, submit, or track a FINTRAC filing; continue in the institution's "
                "approved reporting workflow."
            )
        else:
            st.success("Human review concluded with RGS not reached. No filing action was created.")
        st.stop()
    
    # Store the narrative state in st.session_state to persist across button clicks/renders
    if "narrative" not in st.session_state:
        st.session_state.narrative = None
    if "narrative_txn" not in st.session_state:
        st.session_state.narrative_txn = None
        
    # If the transaction changes, clear the cached narrative
    if st.session_state.narrative_txn != selected_txn['trans_num']:
        st.session_state.narrative = None
        st.session_state.narrative_txn = selected_txn['trans_num']
        
    btn_col, status_col = st.columns([1, 3])
    
    with btn_col:
        generate_btn = st.button("Generate Narrative Draft 🚀", use_container_width=True)
        
    with status_col:
        s3_bucket = os.environ.get("COMPLIANCE_S3_BUCKET")
        if s3_bucket:
            st.markdown(f"🔒 **Configured Evidence Archive:** S3 Bucket `{s3_bucket}`")
        else:
            st.markdown(
                "ℹ️ **Remote Evidence Archive:** Not configured "
                "(`COMPLIANCE_S3_BUCKET` missing); local evidence remains hash-verifiable."
            )
            
    if generate_btn:
        with st.spinner("Invoking narrative drafting assistant (AWS Bedrock / Claude)..."):
            narrative = generate_sar_narrative(
                selected_txn, p_m2_val, p_m3_val, p_m4_val, meta_score
            )
            if narrative:
                st.session_state.narrative = narrative
                st.success("Investigation narrative draft generated for human review.")
            else:
                st.error("Failed to generate a narrative draft. Please verify AWS credentials and Bedrock runtime access.")
                
    if st.session_state.narrative:
        st.markdown("#### Preview 5W+H Investigation Narrative (Draft — Not Filed)")
        
        # Render narrative using the parsed HTML/Markdown layout cards
        render_parsed_compliance_report(st.session_state.narrative)
        
        # Action button to save a draft artifact. This is not a FINTRAC submission.
        st.write("")
        if st.button("Validate & Save Draft 🔒"):
            with st.spinner("Checking factual-grounding guardrails and saving the draft..."):
                try:
                    file_path = save_sar_report(
                        selected_txn,
                        p_m2_val,
                        p_m3_val,
                        p_m4_val,
                        meta_score,
                        st.session_state.narrative,
                        case_id=case_record.case_id,
                        actor=reviewer_id,
                        actor_role=reviewer_role,
                        expected_case_version=case_record.version,
                        case_store=case_store,
                    )
                    saved = case_store.list_evidence(case_record.case_id)[-1]
                    st.success(
                        f"Draft preserved and linked as `{saved.evidence_id}` at `{file_path}`. "
                        f"SHA-256: `{saved.sha256}`"
                    )
                    if saved.archive_receipt and saved.archive_receipt.get("verified"):
                        archive = saved.archive_receipt
                        st.success(
                            f"S3 receipt verified: version `{archive['version_id']}`, "
                            f"request `{archive.get('request_id')}`."
                        )
                    elif saved.archive_receipt:
                        st.warning(
                            "The local evidence is linked, but remote archival is unverified: "
                            f"{saved.archive_receipt.get('error')}"
                        )
                except ValueError as ve:
                    st.error(f"Draft validation failed: {ve}")
                except Exception as ex:
                    st.error(f"Failed to save the narrative draft: {ex}")
else:
    st.success(
        "This transaction does not exceed the model's alert threshold. No model alert "
        "was created; this is not a regulatory determination."
    )
