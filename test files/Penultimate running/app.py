"""
app.py  -  Intrusion Detection System | Streamlit UI
Run with:  streamlit run app.py
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import predict_connection

st.set_page_config(
    page_title="Network IDS",
    page_icon="shield",
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

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.2rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #f1f5f9;
}

label { font-size: 0.82rem !important; color: #475569 !important; font-weight: 500 !important; }

/* Custom stepper widget */
.stepper-label {
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
    margin-bottom: 2px;
}
.stepper-row {
    display: flex;
    align-items: center;
    gap: 0;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
    background: #fff;
    height: 38px;
    margin-bottom: 10px;
}
.stepper-val {
    flex: 1;
    text-align: center;
    font-size: 0.9rem;
    font-weight: 600;
    color: #0f172a;
    background: #f8fafc;
    padding: 0 4px;
    border-left: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    line-height: 38px;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Stepper buttons — kept separate from stButton to avoid overriding the Analyze button */
div[data-testid="stHorizontalBlock"] > div:first-child button,
div[data-testid="stHorizontalBlock"] > div:last-child button {
    /* intentionally unstyled — overriding these caused the breakage */
}

[data-testid="stSelectbox"] { border-radius: 6px; }

.main .block-container { padding: 2rem 2.5rem; max-width: 860px; }

.page-title  { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem; }
.page-subtitle { font-size: 0.88rem; color: #64748b; margin-bottom: 2rem; }

.result-good {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-radius: 12px; padding: 2rem; text-align: center;
}
.result-bad {
    background: #fff1f2; border: 1.5px solid #fca5a5;
    border-radius: 12px; padding: 2rem; text-align: center;
}
.result-empty {
    background: #f8fafc; border: 1.5px dashed #cbd5e1;
    border-radius: 12px; padding: 2.5rem; text-align: center; color: #94a3b8;
}
.verdict-text { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
.verdict-good { color: #16a34a; }
.verdict-bad  { color: #dc2626; }
.verdict-desc { font-size: 0.9rem; color: #64748b; margin-bottom: 1.5rem; }
.conf-track   { background: #e2e8f0; border-radius: 999px; height: 8px; overflow: hidden; margin-bottom: 0.4rem; }
.conf-fill-good { background: #22c55e; height: 100%; border-radius: 999px; }
.conf-fill-bad  { background: #ef4444; height: 100%; border-radius: 999px; }
.conf-label   { font-size: 0.78rem; color: #64748b; margin-bottom: 0.3rem; }
.votes-grid   { display: flex; gap: 0.6rem; justify-content: center; margin-top: 1.2rem; }
.vote-chip    { flex: 1; max-width: 120px; border-radius: 8px; padding: 0.6rem 0.5rem; text-align: center; font-size: 0.78rem; }
.vote-chip-name { color: #64748b; margin-bottom: 0.2rem; font-weight: 500; }
.vote-chip-val  { font-weight: 700; font-size: 0.88rem; }
.chip-good { background: #f0fdf4; border: 1px solid #86efac; }
.chip-good .vote-chip-val { color: #16a34a; }
.chip-bad  { background: #fff1f2; border: 1px solid #fca5a5; }
.chip-bad  .vote-chip-val { color: #dc2626; }
.info-box  { background: #f1f5f9; border-radius: 8px; padding: 0.9rem 1rem; font-size: 0.82rem; color: #475569; margin-top: 1rem; line-height: 1.6; }
.divider-label { font-size: 0.75rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.06em; text-transform: uppercase; margin: 1.4rem 0 0.8rem; }

/* Fix st.metric — value & label invisible (white-on-white) in light theme */
[data-testid="stMetric"]                { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"] p         { color: #64748b !important; font-size: 0.78rem !important; font-weight: 500 !important; }
[data-testid="stMetricValue"]           { color: #0f172a !important; }
[data-testid="stMetricValue"] > div     { color: #0f172a !important; font-size: 1.1rem !important; font-weight: 700 !important; }

/* Only target the main Analyze button, not the stepper buttons */
div.analyze-btn button {
    background-color: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: custom stepper (replaces st.number_input) ────────────────────────
def stepper(label: str, key: str, default: int = 0, min_val: int = 0, max_val: int = 999999):
    """Renders a label + [−] [value] [+] row using session_state."""
    if key not in st.session_state:
        st.session_state[key] = default

    st.markdown(f'<div class="stepper-label">{label}</div>', unsafe_allow_html=True)
    c_minus, c_val, c_plus = st.columns([1, 2, 1])

    with c_minus:
        if st.button("−", key=key + "_minus", use_container_width=True):
            if st.session_state[key] > min_val:
                st.session_state[key] -= 1

    with c_val:
        st.markdown(
            f'<div class="stepper-val">{st.session_state[key]}</div>',
            unsafe_allow_html=True
        )

    with c_plus:
        if st.button("+", key=key + "_plus", use_container_width=True):
            if st.session_state[key] < max_val:
                st.session_state[key] += 1

    return st.session_state[key]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### IDS Input Panel")
    st.caption("Fill in the connection details, then click **Analyze** on the right.")

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
    duration = stepper("Duration (seconds)", "duration", default=0)

    st.markdown('<div class="sidebar-section">Data Transfer</div>', unsafe_allow_html=True)
    src_bytes = stepper("Bytes Sent (src to dst)",     "src_bytes", default=215)
    dst_bytes = stepper("Bytes Received (dst to src)", "dst_bytes", default=45076)

    st.markdown('<div class="sidebar-section">Login & Access</div>', unsafe_allow_html=True)
    logged_in      = st.selectbox("User Logged In?",    [1, 0], format_func=lambda x: "Yes" if x else "No", key="logged_in")
    is_guest_login = st.selectbox("Guest Login?",       [0, 1], format_func=lambda x: "Yes" if x else "No", key="is_guest_login")
    is_host_login  = st.selectbox("Host Login?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="is_host_login")
    land           = st.selectbox("Same src/dst host?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="land")

    st.markdown('<div class="sidebar-section">Suspicious Activity</div>', unsafe_allow_html=True)
    root_shell         = st.selectbox("Root Shell Obtained?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="root_shell")
    su_attempted       = st.selectbox("SU Attempted?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="su_attempted")
    num_failed_logins  = stepper("Failed Login Attempts",    "num_failed_logins",  default=0)
    num_compromised    = stepper("Compromised Conditions",   "num_compromised",    default=0)
    num_root           = stepper("Root Access Count",        "num_root",           default=0)
    num_shells         = stepper("Shell Prompts",            "num_shells",         default=0)
    num_file_creations = stepper("Files Created",            "num_file_creations", default=0)
    num_access_files   = stepper("Sensitive Files Accessed", "num_access_files",   default=0)
    num_outbound_cmds  = stepper("Outbound Commands",        "num_outbound_cmds",  default=0)

    st.markdown('<div class="sidebar-section">Traffic Statistics</div>', unsafe_allow_html=True)
    hot            = stepper("Hot Indicators",                   "hot",       default=0)
    wrong_fragment = stepper("Wrong Fragments",                  "wrong_fragment", default=0)
    urgent         = stepper("Urgent Packets",                   "urgent",    default=0)
    count          = stepper("Connections to Same Host (2s)",    "count",     default=1)
    srv_count      = stepper("Connections to Same Service (2s)", "srv_count", default=1)
    serror_rate        = st.slider("SYN Error Rate",         0.0, 1.0, 0.0, key="serror_rate")
    srv_serror_rate    = st.slider("Srv SYN Error Rate",     0.0, 1.0, 0.0, key="srv_serror_rate")
    rerror_rate        = st.slider("REJ Error Rate",         0.0, 1.0, 0.0, key="rerror_rate")
    srv_rerror_rate    = st.slider("Srv REJ Error Rate",     0.0, 1.0, 0.0, key="srv_rerror_rate")
    same_srv_rate      = st.slider("Same Service Rate",      0.0, 1.0, 1.0, key="same_srv_rate")
    diff_srv_rate      = st.slider("Different Service Rate", 0.0, 1.0, 0.0, key="diff_srv_rate")
    srv_diff_host_rate = st.slider("Srv Diff Host Rate",     0.0, 1.0, 0.0, key="srv_diff_host_rate")

    st.markdown('<div class="sidebar-section">Destination Host Stats</div>', unsafe_allow_html=True)
    dst_host_count     = stepper("DH Count",         "dst_host_count",     default=255, max_val=255)
    dst_host_srv_count = stepper("DH Service Count", "dst_host_srv_count", default=11,  max_val=255)
    dst_host_same_srv_rate      = st.slider("DH Same Service Rate",  0.0, 1.0, 0.04, key="dst_host_same_srv_rate")
    dst_host_diff_srv_rate      = st.slider("DH Diff Service Rate",  0.0, 1.0, 0.06, key="dst_host_diff_srv_rate")
    dst_host_same_src_port_rate = st.slider("DH Same Src Port Rate", 0.0, 1.0, 0.0,  key="dst_host_same_src_port_rate")
    dst_host_srv_diff_host_rate = st.slider("DH Srv Diff Host Rate", 0.0, 1.0, 0.0,  key="dst_host_srv_diff_host_rate")
    dst_host_serror_rate        = st.slider("DH SYN Error Rate",     0.0, 1.0, 0.0,  key="dst_host_serror_rate")
    dst_host_srv_serror_rate    = st.slider("DH Srv SYN Error Rate", 0.0, 1.0, 0.0,  key="dst_host_srv_serror_rate")
    dst_host_rerror_rate        = st.slider("DH REJ Error Rate",     0.0, 1.0, 0.0,  key="dst_host_rerror_rate")
    dst_host_srv_rerror_rate    = st.slider("DH Srv REJ Error Rate", 0.0, 1.0, 0.0,  key="dst_host_srv_rerror_rate")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Network Intrusion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Enter connection details in the sidebar, then run the analysis.</div>', unsafe_allow_html=True)

col_btn, col_gap = st.columns([1, 2])
with col_btn:
    with st.container():
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        predict_clicked = st.button("Analyze Connection", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Result ────────────────────────────────────────────────────────────────────
if not predict_clicked:
    st.markdown(
        '<div class="result-empty">'
        '<div style="font-size:2.2rem;margin-bottom:0.6rem">&#128269;</div>'
        '<div style="font-weight:600;color:#475569;margin-bottom:0.3rem">No analysis yet</div>'
        '<div style="font-size:0.85rem">Fill in the sidebar and click <strong>Analyze Connection</strong></div>'
        '</div>',
        unsafe_allow_html=True
    )
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
            fill_class = "conf-fill-good" if is_good else "conf-fill-bad"
            icon_html  = "&#9989;" if is_good else "&#9888;"
            desc       = ("This connection looks normal and safe."
                          if is_good else
                          "This connection shows signs of an intrusion attempt.")

            def make_chip(label, vote):
                c = "chip-good" if vote == "GOOD" else "chip-bad"
                v = "&#10003; Good" if vote == "GOOD" else "&#10007; Bad"
                return (
                    '<div class="vote-chip ' + c + '">'
                    '<div class="vote-chip-name">' + label + '</div>'
                    '<div class="vote-chip-val">' + v + '</div>'
                    '</div>'
                )

            chips = (
                make_chip("Naive Bayes",   votes["GNB"])
                + make_chip("Decision Tree", votes["DT"])
                + make_chip("XGBoost",       votes["XGB"])
            )

            html = (
                '<div class="' + card_class + '">'
                + '<div class="verdict-text ' + verd_class + '">' + icon_html + '&nbsp;' + result + ' Connection</div>'
                + '<div class="verdict-desc">' + desc + '</div>'
                + '<div class="conf-label">Model agreement &mdash; ' + str(confidence) + '%</div>'
                + '<div class="conf-track"><div class="' + fill_class + '" style="width:' + str(confidence) + '%;height:100%;border-radius:999px"></div></div>'
                + '<div class="votes-grid">' + chips + '</div>'
                + '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
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

            st.markdown(
                '<div class="info-box"><strong>How this works:</strong> Three different models '
                '(Gaussian Naive Bayes, Decision Tree, and XGBoost) each independently analyze '
                'the connection. The final result is decided by majority vote — if 2 or more '
                'models agree, that is the prediction. A 100% confidence means all 3 agreed.</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"**Prediction failed:** {e}")
            st.info("Make sure the `ids_model/` folder is in the same directory as `app.py` and `predictor.py`.")
