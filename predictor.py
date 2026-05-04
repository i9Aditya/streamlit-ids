import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

# =========================================================
# DUMMY CLASSES (FIX FOR PICKLE ERROR)
# Injected into __main__ BEFORE joblib.load() so pickle
# can resolve them regardless of which file is the entry point.
# =========================================================

class Decision_Node:
    pass

class Leaf:
    pass

class Question:
    pass

import __main__
for _cls in (Decision_Node, Leaf, Question):
    if not hasattr(__main__, _cls.__name__):
        setattr(__main__, _cls.__name__, _cls)

# =========================================================
# CATEGORICAL ENCODINGS
# The Decision Tree was trained on the full NSL-KDD feature
# vector where protocol_type / service / flag were label-encoded
# as integers. We must replicate that encoding here so that
# _Question.match() never receives a string.
# =========================================================

_PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

_SERVICE_MAP = {
    "aol": 0, "auth": 1, "bgp": 2, "courier": 3, "csnet_ns": 4,
    "ctf": 5, "daytime": 6, "discard": 7, "domain": 8, "domain_u": 9,
    "echo": 10, "eco_i": 11, "efs": 12, "exec": 13, "finger": 14,
    "ftp": 15, "ftp_data": 16, "gopher": 17, "harvest": 18,
    "hostnames": 19, "http": 20, "http_443": 21, "http_8001": 22,
    "icmp": 23, "imap4": 24, "IRC": 25, "iso_tsap": 26, "klogin": 27,
    "kshell": 28, "ldap": 29, "link": 30, "login": 31, "mtp": 32,
    "name": 33, "netbios_dgm": 34, "netbios_ns": 35, "netbios_ssn": 36,
    "netstat": 37, "nnsp": 38, "ntp_u": 39, "other": 40, "pm_dump": 41,
    "pop_2": 42, "pop_3": 43, "printer": 44, "private": 45,
    "red_i": 46, "remote_job": 47, "rje": 48, "shell": 49,
    "smtp": 50, "sql_net": 51, "ssh": 52, "sunrpc": 53, "supdup": 54,
    "systat": 55, "telnet": 56, "time": 57, "tim_i": 58, "urh_i": 59,
    "urp_i": 60, "uucp": 61, "uucp_path": 62, "vmnet": 63,
    "whois": 64, "worm": 65, "X11": 66, "Z39_50": 67, "charter": 68,
}

_FLAG_MAP = {
    "SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "RSTR": 4,
    "S1": 5, "S2": 6, "S3": 7, "OTH": 8, "SH": 9, "RSTOS0": 10,
}

# Full NSL-KDD column order the DT was trained on (41 features)
_DT_FEATURE_ORDER = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

def _encode_for_dt(input_dict):
    """Return a fully numeric list in the original DT training feature order."""
    row = []
    for col in _DT_FEATURE_ORDER:
        val = input_dict.get(col, 0)
        if col == "protocol_type":
            val = _PROTOCOL_MAP.get(str(val).lower(), 0)
        elif col == "service":
            val = _SERVICE_MAP.get(str(val), 20)
        elif col == "flag":
            val = _FLAG_MAP.get(str(val), 0)
        else:
            val = float(val)
        row.append(val)
    row.append(0)   # dummy label slot expected by _classify
    return row

# =========================================================
# GNB MODEL
# BUG FIX 1: The original predict() applied a SINGLE prior
# value to EVERY class — meaning all classes were multiplied
# by the same number, making the prior useless and biasing
# results toward whichever class had higher likelihoods
# (almost always the attack/BAD class due to data imbalance).
# Fix: store per-class priors and apply the correct one.
# =========================================================

class _GNBPredictor:
    def __init__(self, data: dict):
        self.mean      = data["mean"]
        self.variance  = data["variance"]
        self.classes   = data["classes"]
        self.n_class   = data["n_class"]

        # Support both old format (single float) and new format (list of priors)
        raw_prior = data["prior"]
        if isinstance(raw_prior, (list, np.ndarray)):
            self.priors = list(raw_prior)
        else:
            # Old pkl stores only prior for class 1 (GOOD).
            # Reconstruct both priors assuming binary classification.
            p_good = float(raw_prior)
            p_bad  = 1.0 - p_good
            # classes is typically [0, 1] → [BAD, GOOD]
            self.priors = [p_bad if c == 0 else p_good for c in self.classes]

    def _gnb_base(self, x_val, x_mean, x_var):
        if x_var == 0:
            x_var = 1e-9
        eq1  = 1.0 / np.sqrt(2 * np.pi * x_var)
        expo = np.exp(-((x_val - x_mean) ** 2) / (2 * x_var))
        return eq1 * expo

    def predict(self, X):
        mean_var = []
        for i in range(self.n_class):
            for j in range(len(self.mean[i])):
                mean_var.append([self.mean[i][j], self.variance[i][j]])

        split       = np.vsplit(np.array(mean_var), self.n_class)
        final_probs = []

        for idx, cls_mv in enumerate(split):
            # BUG FIX 1: use per-class prior, not a single shared value
            likelihood = np.prod([
                self._gnb_base(X[j], cls_mv[j][0], cls_mv[j][1])
                for j in range(len(cls_mv))
            ])
            # Use log-space to avoid floating point underflow on many features
            log_prob = np.sum([
                np.log(max(self._gnb_base(X[j], cls_mv[j][0], cls_mv[j][1]), 1e-300))
                for j in range(len(cls_mv))
            ]) + np.log(max(self.priors[idx], 1e-300))
            final_probs.append(log_prob)

        return self.classes[final_probs.index(max(final_probs))]

# =========================================================
# DECISION TREE HELPERS
# =========================================================

def _is_numeric(value):
    return isinstance(value, (int, float))

def _to_float(value):
    try:
        return float(value)
    except:
        return None

class _Question:
    def __init__(self, column, value):
        self.column = column
        self.value  = value

    # def match(self, example):
    #     val = example[self.column]
    #     if _is_numeric(val):
    #         return float(val) >= float(self.value)
    #     return val == self.value
    
    
    def match(self, example):
        val = example[self.column]

        val_num = _to_float(val)
        value_num = _to_float(self.value)

        # If BOTH are numeric → compare
        if val_num is not None and value_num is not None:
            return val_num >= value_num

        # Otherwise → treat as categorical
            return str(val) == str(self.value)

class _Leaf:
    def __init__(self, rows):
        counts = {}
        for row in rows:
            lbl         = row[-1]
            counts[lbl] = counts.get(lbl, 0) + 1
        self.predictions = counts

class _Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question     = question
        self.true_branch  = true_branch
        self.false_branch = false_branch

def _classify(row, node):
    if isinstance(node, _Leaf):
        return node.predictions
    if node.question.match(row):
        return _classify(row, node.true_branch)
    return _classify(row, node.false_branch)

def _patch_dt(node):
    if node is None:
        return node

    if hasattr(node, "predictions") and not hasattr(node, "question"):
        leaf             = _Leaf.__new__(_Leaf)
        leaf.predictions = node.predictions
        return leaf

    if hasattr(node, "question"):
        q        = node.question
        new_q    = _Question.__new__(_Question)
        new_q.column = q.column
        new_q.value  = q.value

        dn              = _Decision_Node.__new__(_Decision_Node)
        dn.question     = new_q
        dn.true_branch  = _patch_dt(node.true_branch)
        dn.false_branch = _patch_dt(node.false_branch)
        return dn

    return node

# =========================================================
# LOAD MODELS
# =========================================================

_ARTIFACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ids_model")

try:
    _gnb_data       = joblib.load(os.path.join(_ARTIFACTS_PATH, "gnb_model.pkl"))
    _dt_model_raw   = joblib.load(os.path.join(_ARTIFACTS_PATH, "dt_model.pkl"))
    _xgb_model      = joblib.load(os.path.join(_ARTIFACTS_PATH, "xgb_model.pkl"))
    _pca_obj        = joblib.load(os.path.join(_ARTIFACTS_PATH, "pca_obj.pkl"))
    _numeric_cols   = joblib.load(os.path.join(_ARTIFACTS_PATH, "numeric_cols.pkl"))
    _vocab_protocol = joblib.load(os.path.join(_ARTIFACTS_PATH, "vocab_protocol.pkl"))
    _vocab_service  = joblib.load(os.path.join(_ARTIFACTS_PATH, "vocab_service.pkl"))
    _vocab_flag     = joblib.load(os.path.join(_ARTIFACTS_PATH, "vocab_flag.pkl"))

    _gnb      = _GNBPredictor(_gnb_data)
    _dt_model = _patch_dt(_dt_model_raw)

    _LOADED = True

except Exception as e:
    _LOADED     = False
    _LOAD_ERROR = str(e)

# =========================================================
# MAIN FUNCTION
# BUG FIX 2: DT vote was reading max() of raw prediction
# counts dict — but the dict keys are class labels (0 or 1)
# and max() on a dict returns the max KEY, not the most
# frequent class. Fixed to use max by value (frequency).
#
# BUG FIX 3: np.bincount requires non-negative integers.
# If any model returns an unexpected value, bincount crashes
# silently or misindexes. Added explicit int casting + guard.
# =========================================================

def predict_connection(input_dict: dict):
    if not _LOADED:
        raise RuntimeError(f"Model load failed: {_LOAD_ERROR}")

    input_df = pd.DataFrame([input_dict])

    # --- GNB ---
    gnb_numeric = input_df[_numeric_cols].values.astype(float)
    gnb_pca     = _pca_obj.transform(gnb_numeric)
    gnb_vote    = int(_gnb.predict(gnb_pca[0]))

    # --- Decision Tree ---
    dt_input_row = _encode_for_dt(input_dict)
    dt_raw       = _classify(dt_input_row, _dt_model)

    # BUG FIX 2: max(dict, key=dict.get) returns the class with
    # the HIGHEST COUNT (most votes), not the highest key value.
    dt_vote = int(max(dt_raw, key=lambda k: dt_raw[k]))

    # --- XGBoost ---
    xgb_pca    = _pca_obj.transform(input_df[_numeric_cols].values.astype(float))
    xgb_pca_df = pd.DataFrame(xgb_pca)

    p_vec = CountVectorizer(vocabulary=_vocab_protocol, binary=True).fit_transform(
        [str(input_dict.get("protocol_type", "tcp"))]
    )
    s_vec = CountVectorizer(vocabulary=_vocab_service, binary=True).fit_transform(
        [str(input_dict.get("service", "http"))]
    )
    f_vec = CountVectorizer(vocabulary=_vocab_flag, binary=True).fit_transform(
        [str(input_dict.get("flag", "SF"))]
    )

    xgb_input = pd.DataFrame(
        hstack((xgb_pca_df, p_vec, s_vec, f_vec)).toarray()
    )
    xgb_input.columns = [str(i) for i in range(xgb_input.shape[1])]
    xgb_vote = int(_xgb_model.predict(xgb_input)[0])

    # --- Majority Voting ---
    # BUG FIX 3: ensure all votes are 0 or 1 before bincount
    votes = [max(0, min(1, v)) for v in [gnb_vote, dt_vote, xgb_vote]]
    final_vote = int(np.bincount(votes).argmax())
    confidence = int(votes.count(final_vote) / 3 * 100)

    result = "GOOD" if final_vote == 1 else "BAD"

    return result, confidence, {
        "GNB": "GOOD" if gnb_vote == 1 else "BAD",
        "DT":  "GOOD" if dt_vote  == 1 else "BAD",
        "XGB": "GOOD" if xgb_vote == 1 else "BAD",
    }
