# ============================================================
# CELL TO ADD AT THE END OF YOUR IDS.ipynb
# ============================================================
# This cell:
#   1. Re-saves all artifacts (with correct scaler for GNB)
#   2. Tests the predict_connection() function inline
#   3. Tells you how to launch the Streamlit app
# ============================================================

import joblib, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

os.makedirs("ids_model", exist_ok=True)

# ── Load clean dataset ──────────────────────────────────────
df_raw = pd.read_pickle("clean_dataset.pkl")
numeric_cols = list(df_raw.select_dtypes(include=['float64','int64']).columns)

# ── Save models ─────────────────────────────────────────────
joblib.dump(gaussian_classifier, "ids_model/gnb_model.pkl")
joblib.dump(my_tree,             "ids_model/dt_model.pkl")
joblib.dump(model,               "ids_model/xgb_model.pkl")
joblib.dump(pca_obj,             "ids_model/pca_obj.pkl")
joblib.dump(numeric_cols,        "ids_model/numeric_cols.pkl")

# ── Save vocabularies ────────────────────────────────────────
joblib.dump(list(set(df_raw['protocol_type'].values)), "ids_model/vocab_protocol.pkl")
joblib.dump(list(set(df_raw['service'].values)),       "ids_model/vocab_service.pkl")
joblib.dump(list(set(df_raw['flag'].values)),          "ids_model/vocab_flag.pkl")

print("✅ All artifacts saved.\n")

# ── Quick inline test of the prediction pipeline ─────────────
# We grab one real sample from the original dataset to verify

sample_row = df_raw.iloc[0]  # first row from clean dataset

input_dict = {col: sample_row[col] for col in df_raw.columns if col != 'intrusion_type'}

print(f"Sample connection type in dataset: {sample_row['intrusion_type']}")
print(f"Sample protocol: {sample_row['protocol_type']}, service: {sample_row['service']}\n")

# ── Inline predict (mirrors predictor.py logic exactly) ──────

def _classify_inline(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return _classify_inline(row, node.true_branch)
    else:
        return _classify_inline(row, node.false_branch)

def quick_predict(input_dict):
    vp = joblib.load("ids_model/vocab_protocol.pkl")
    vs = joblib.load("ids_model/vocab_service.pkl")
    vf = joblib.load("ids_model/vocab_flag.pkl")
    nc = joblib.load("ids_model/numeric_cols.pkl")
    pc = joblib.load("ids_model/pca_obj.pkl")
    
    df_in = pd.DataFrame([input_dict])

    # GNB
    gnb_pca  = pc.transform(df_in[nc].values.astype(float))
    gnb_vote = int(gaussian_classifier.predict(gnb_pca[0]))

    # DT
    FEATURE_ORDER = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate'
    ]
    dt_row  = [input_dict.get(col, 0) for col in FEATURE_ORDER] + [0]
    dt_raw  = _classify_inline(dt_row, my_tree)
    dt_vote = int(sorted(dt_raw.items(), key=lambda x: x[1], reverse=True)[0][0])

    # XGB
    xgb_pca = pc.transform(df_in[nc].values.astype(float))
    xgb_df  = pd.DataFrame(xgb_pca)
    p_vec   = CountVectorizer(vocabulary=vp, binary=True).fit_transform([str(input_dict.get('protocol_type','tcp'))])
    s_vec   = CountVectorizer(vocabulary=vs, binary=True).fit_transform([str(input_dict.get('service','http'))])
    f_vec   = CountVectorizer(vocabulary=vf, binary=True).fit_transform([str(input_dict.get('flag','SF'))])
    xgb_in  = pd.DataFrame(hstack((xgb_df, p_vec, s_vec, f_vec)).toarray())
    xgb_vote = int(model.predict(xgb_in)[0])

    votes      = [gnb_vote, dt_vote, xgb_vote]
    final      = int(np.bincount(votes).argmax())
    confidence = int(votes.count(final)/3*100)
    result     = "GOOD ✅" if final == 1 else "BAD ⚠️"

    print(f"GNB vote : {'GOOD' if gnb_vote==1 else 'BAD'}")
    print(f"DT  vote : {'GOOD' if dt_vote ==1 else 'BAD'}")
    print(f"XGB vote : {'GOOD' if xgb_vote==1 else 'BAD'}")
    print(f"─────────────────────────")
    print(f"Result   : {result}")
    print(f"Confidence: {confidence}%")

quick_predict(input_dict)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀  HOW TO LAUNCH THE STREAMLIT APP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Make sure these 3 things are in the SAME folder:
     ├── app.py
     ├── predictor.py
     └── ids_model/          ← (the folder we just saved)

2. Open Command Prompt in that folder and run:
     streamlit run app.py

3. Your browser will open automatically at:
     http://localhost:8501
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
