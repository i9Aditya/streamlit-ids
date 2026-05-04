"""
app.py  —  Intrusion Detection System | Streamlit UI
Run with:  streamlit run app.py
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import predict_connection

st.set_page_config(
    page_title="Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
}

#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* Sidebar section headers */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #f1f5f9;
}

/* Labels */
label { font-size: 0.82rem !important; color: #475569 !important; font-weight: 500 !important; }

/* Number inputs — style only the input box, never touch selectbox internals */
[data-testid="stNumberInput"] input {
    border-radius: 6px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 0.88rem !important;
    background: #f8fafc !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/*
 * FIX: Do NOT apply background or border overrides to stSelectbox inner divs.
 * The old rule `[data-testid="stSelectbox"] > div > div { background: #f8fafc !important }`
 * was painting over Streamlit's dropdown portal/overlay, making selections invisible.
 * Only style the outer wrapper for a subtle border — leave all inner elements alone.
 */
[data-testid="stSelectbox"] {
    border-radius: 6px;
}

/* Main content area */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 860px;
}

/* Page title */
.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 2rem;
}

/* Result cards */
.result-good {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-bad {
    background: #fff1f2;
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-empty {
    background: #f8fafc;
    border: 1.5px dashed #cbd5e1;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    color: #94a3b8;
}

.verdict-text {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.verdict-good { color: #16a34a; }
.verdict-bad  { color: #dc2626; }
.verdict-desc { font-size: 0.9rem; color: #64748b; margin-bottom: 1.5rem; }

/* Confidence bar */
.conf-track {
    background: #e2e8f0;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 0.4rem;
}
.conf-fill-good { background: #22c55e; height: 100%; border-radius: 999px; }
.conf-fill-bad  { background: #ef4444; height: 100%; border-radius: 999px; }
.conf-label { font-size: 0.78rem; color: #64748b; margin-bottom: 0.3rem; }
.conf-pct   { font-size: 0.9rem; font-weight: 600; }

/* Vote chips */
.votes-grid {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    margin-top: 1.2rem;
}
.vote-chip {
    flex: 1;
    max-width: 120px;
    border-radius: 8px;
    padding: 0.6rem 0.5rem;
    text-align: center;
    font-size: 0.78rem;
}
.vote-chip-name { color: #64748b; margin-bottom: 0.2rem; font-weight: 500; }
.vote-chip-val  { font-weight: 700; font-size: 0.88rem; }
.chip-good { background: #f0fdf4; border: 1px solid #86efac; }
.chip-good .vote-chip-val { color: #16a34a; }
.chip-bad  { background: #fff1f2; border: 1px solid #fca5a5; }
.chip-bad  .vote-chip-val { color: #dc2626; }

/* Info box */
.info-box {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    font-size: 0.82rem;
    color: #475569;
    margin-top: 1rem;
    line-height: 1.6;
}

/* Divider label */
.divider-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.8rem;
}

/* Predict button */
[data-testid="stButton"] > button {
    background-color: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    width: 100%;
    transition: background 0.15s ease !important;
}
[data-testid="stButton"] > button:hover {
    background-color: #4f46e5 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar — all inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ IDS Input Panel")
    st.caption("Fill in the connection details below, then click **Analyze** on the right.")

    st.markdown('<div class="sidebar-section">Basic Info</div>', unsafe_allow_html=True)
    protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"], key="protocol_type")
    service = st.selectbox("Service", [
        "http","ftp","smtp","ssh","telnet","domain","ftp_data","other","private",
        "eco_i","mtp","finger","domain_u","supdup","uucp","csnet_ns","pop_3",
        "sunrpc","auth","courier","ctf","whois","login","imap4","IRC","netstat",
        "nnsp","urp_i","X11","Z39_50","aol","bgp","daytime","discard","echo",
        "efs","exec","gopher","hostnames","http_443","http_8001","icmp","iso_tsap",
        "klogin","kshell","ldap","link","name","netbios_dgm","netbios_ns",
        "netbios_ssn","ntp_u","pm_dump","pop_2","printer","red_i","remote_job",
        "rje","shell","sql_net","systat","time","tim_i","urh_i","uucp_path",
        "vmnet","worm","harvest","charter"
    ], key="service")
    flag     = st.selectbox("Connection Flag", ["SF","S0","REJ","RSTO","RSTR","S1","S2","S3","OTH","SH","RSTOS0"], key="flag")
    duration = st.number_input("Duration (seconds)", min_value=0, value=0, key="duration")

    st.markdown('<div class="sidebar-section">Data Transfer</div>', unsafe_allow_html=True)
    src_bytes = st.number_input("Bytes Sent (src → dst)",      min_value=0, value=215,   key="src_bytes")
    dst_bytes = st.number_input("Bytes Received (dst → src)",  min_value=0, value=45076, key="dst_bytes")

    st.markdown('<div class="sidebar-section">Login & Access</div>', unsafe_allow_html=True)
    logged_in      = st.selectbox("User Logged In?",    [1, 0], format_func=lambda x: "Yes" if x else "No", key="logged_in")
    is_guest_login = st.selectbox("Guest Login?",       [0, 1], format_func=lambda x: "Yes" if x else "No", key="is_guest_login")
    is_host_login  = st.selectbox("Host Login?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="is_host_login")
    land           = st.selectbox("Same src/dst host?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="land")

    st.markdown('<div class="sidebar-section">Suspicious Activity</div>', unsafe_allow_html=True)
    root_shell         = st.selectbox("Root Shell Obtained?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="root_shell")
    su_attempted       = st.selectbox("SU Attempted?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="su_attempted")
    num_failed_logins  = st.number_input("Failed Login Attempts",    min_value=0, value=0, key="num_failed_logins")
    num_compromised    = st.number_input("Compromised Conditions",   min_value=0, value=0, key="num_compromised")
    num_root           = st.number_input("Root Access Count",        min_value=0, value=0, key="num_root")
    num_shells         = st.number_input("Shell Prompts",            min_value=0, value=0, key="num_shells")
    num_file_creations = st.number_input("Files Created",            min_value=0, value=0, key="num_file_creations")
    num_access_files   = st.number_input("Sensitive Files Accessed", min_value=0, value=0, key="num_access_files")
    num_outbound_cmds  = st.number_input("Outbound Commands",        min_value=0, value=0, key="num_outbound_cmds")

    st.markdown('<div class="sidebar-section">Traffic Statistics</div>', unsafe_allow_html=True)
    hot            = st.number_input("Hot Indicators",                   min_value=0, value=0, key="hot")
    wrong_fragment = st.number_input("Wrong Fragments",                  min_value=0, value=0, key="wrong_fragment")
    urgent         = st.number_input("Urgent Packets",                   min_value=0, value=0, key="urgent")
    count          = st.number_input("Connections to Same Host (2s)",    min_value=0, value=1, key="count")
    srv_count      = st.number_input("Connections to Same Service (2s)", min_value=0, value=1, key="srv_count")
    serror_rate        = st.slider("SYN Error Rate",         0.0, 1.0, 0.0, key="serror_rate")
    srv_serror_rate    = st.slider("Srv SYN Error Rate",     0.0, 1.0, 0.0, key="srv_serror_rate")
    rerror_rate        = st.slider("REJ Error Rate",         0.0, 1.0, 0.0, key="rerror_rate")
    srv_rerror_rate    = st.slider("Srv REJ Error Rate",     0.0, 1.0, 0.0, key="srv_rerror_rate")
    same_srv_rate      = st.slider("Same Service Rate",      0.0, 1.0, 1.0, key="same_srv_rate")
    diff_srv_rate      = st.slider("Different Service Rate", 0.0, 1.0, 0.0, key="diff_srv_rate")
    srv_diff_host_rate = st.slider("Srv Diff Host Rate",     0.0, 1.0, 0.0, key="srv_diff_host_rate")

    st.markdown('<div class="sidebar-section">Destination Host Stats</div>', unsafe_allow_html=True)
    dst_host_count              = st.number_input("DH Count",         min_value=0, max_value=255, value=255, key="dst_host_count")
    dst_host_srv_count          = st.number_input("DH Service Count", min_value=0, max_value=255, value=11,  key="dst_host_srv_count")
    dst_host_same_srv_rate      = st.slider("DH Same Service Rate",  0.0, 1.0, 0.04, key="dst_host_same_srv_rate")
    dst_host_diff_srv_rate      = st.slider("DH Diff Service Rate",  0.0, 1.0, 0.06, key="dst_host_diff_srv_rate")
    dst_host_same_src_port_rate = st.slider("DH Same Src Port Rate", 0.0, 1.0, 0.0,  key="dst_host_same_src_port_rate")
    dst_host_srv_diff_host_rate = st.slider("DH Srv Diff Host Rate", 0.0, 1.0, 0.0,  key="dst_host_srv_diff_host_rate")
    dst_host_serror_rate        = st.slider("DH SYN Error Rate",     0.0, 1.0, 0.0,  key="dst_host_serror_rate")
    dst_host_srv_serror_rate    = st.slider("DH Srv SYN Error Rate", 0.0, 1.0, 0.0,  key="dst_host_srv_serror_rate")
    dst_host_rerror_rate        = st.slider("DH REJ Error Rate",     0.0, 1.0, 0.0,  key="dst_host_rerror_rate")
    dst_host_srv_rerror_rate    = st.slider("DH Srv REJ Error Rate", 0.0, 1.0, 0.0,  key="dst_host_srv_rerror_rate")


# ── Main area ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Network Intrusion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Enter connection details in the sidebar, then run the analysis.</div>', unsafe_allow_html=True)

col_btn, col_gap = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("🔍  Analyze Connection")

st.markdown("---")

# ── Result ──────────────────────────────────────────────────────────────────
if not predict_clicked:
    st.markdown("""
    <div class="result-empty">
        <div style="font-size:2.2rem; margin-bottom:0.6rem">🔍</div>
        <div style="font-weight:600; color:#475569; margin-bottom:0.3rem">No analysis yet</div>
        <div style="font-size:0.85rem">Fill in the sidebar and click <strong>Analyze Connection</strong></div>
    </div>
    """, unsafe_allow_html=True)

else:
    input_data = {
        "duration": duration, "protocol_type": protocol_type,
        "service": service, "flag": flag,
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "land": land, "wrong_fragment": wrong_fragment,
        "urgent": urgent, "hot": hot,
        "num_failed_logins": num_failed_logins, "logged_in": logged_in,
        "num_compromised": num_compromised, "root_shell": root_shell,
        "su_attempted": su_attempted, "num_root": num_root,
        "num_file_creations": num_file_creations, "num_shells": num_shells,
        "num_access_files": num_access_files, "num_outbound_cmds": num_outbound_cmds,
        "is_host_login": is_host_login, "is_guest_login": is_guest_login,
        "count": count, "srv_count": srv_count,
        "serror_rate": serror_rate, "srv_serror_rate": srv_serror_rate,
        "rerror_rate": rerror_rate, "srv_rerror_rate": srv_rerror_rate,
        "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
        "srv_diff_host_rate": srv_diff_host_rate,
        "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": dst_host_same_srv_rate,
        "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
        "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
        "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
        "dst_host_serror_rate": dst_host_serror_rate,
        "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
        "dst_host_rerror_rate": dst_host_rerror_rate,
        "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate,
    }

    with st.spinner("Running ensemble analysis..."):
        try:
            result, confidence, votes = predict_connection(input_data)

            is_good    = result == "GOOD"
            card_class = "result-good" if is_good else "result-bad"
            verd_class = "verdict-good" if is_good else "verdict-bad"
            icon       = "✅" if is_good else "⚠️"
            desc       = "This connection looks normal and safe." if is_good else "This connection shows signs of an intrusion attempt."
            fill_class = "conf-fill-good" if is_good else "conf-fill-bad"

            def chip(model, vote):
                cls = "chip-good" if vote == "GOOD" else "chip-bad"
                lbl = "✓ Good" if vote == "GOOD" else "✗ Bad"
                return f"""
                <div class="vote-chip {cls}">
                    <div class="vote-chip-name">{model}</div>
                    <div class="vote-chip-val">{lbl}</div>
                </div>"""

            st.markdown(f"""
            <div class="{card_class}">
                <div class="verdict-text {verd_class}">{icon} &nbsp;{result} Connection</div>
                <div class="verdict-desc">{desc}</div>

                <div class="conf-label">Model agreement — {confidence}%</div>
                <div class="conf-track">
                    <div class="{fill_class}" style="width:{confidence}%"></div>
                </div>

                <div class="votes-grid">
                    {chip("Naive Bayes", votes["GNB"])}
                    {chip("Decision Tree", votes["DT"])}
                    {chip("XGBoost", votes["XGB"])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if is_good and confidence == 100:
                st.success("**All 3 models agree** — this connection is normal. No action needed.")
            elif is_good and confidence == 67:
                st.warning("**2 out of 3 models** say this is normal, but one flagged it. Worth a closer look.")
            elif not is_good and confidence == 67:
                st.error("**2 out of 3 models** flagged this as an attack. Investigate this connection.")
            else:
                st.error("**All 3 models agree** — this is an intrusion attempt. Take immediate action.")

            st.markdown('<div class="divider-label">Connection Summary</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Protocol",       protocol_type.upper())
            c2.metric("Service",        service)
            c3.metric("Bytes Sent",     f"{src_bytes:,}")
            c4.metric("Bytes Received", f"{dst_bytes:,}")

            st.markdown(f"""
            <div class="info-box">
                <strong>How this works:</strong> Three different models (Gaussian Naive Bayes, Decision Tree,
                and XGBoost) each independently analyze the connection. The final result is decided by majority
                vote — if 2 or more models agree, that's the prediction. A 100% confidence means all 3 agreed.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"**Prediction failed:** {e}")
            st.info("Make sure the `ids_model/` folder is in the same directory as `app.py` and `predictor.py`.")
