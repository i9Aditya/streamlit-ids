"""
predictor.py
------------
Reusable prediction pipeline for the IDS (Intrusion Detection System).
GNB is reconstructed from raw data (mean/variance/classes) to avoid
pickle class-lookup errors across different Python modules.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


# ---------------------------------------------------------------------------
# GNB: rebuilt from saved data — no pickle class dependency
# ---------------------------------------------------------------------------

class _GNBPredictor:
    """Lightweight GNB inference wrapper built from saved mean/variance data."""

    def __init__(self, data: dict):
        self.mean      = data["mean"]
        self.variance  = data["variance"]
        self.classes   = data["classes"]
        self.prior     = data["prior"]
        self.n_class   = data["n_class"]

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

        split = np.vsplit(np.array(mean_var), self.n_class)
        final_probs = []
        for cls_mv in split:
            prob = np.prod([self._gnb_base(X[j], cls_mv[j][0], cls_mv[j][1])
                            for j in range(len(cls_mv))]) * self.prior
            final_probs.append(prob)

        return self.classes[final_probs.index(max(final_probs))]


# ---------------------------------------------------------------------------
# Decision Tree helpers — mirror notebook exactly
# ---------------------------------------------------------------------------

def _is_numeric(value):
    return isinstance(value, (int, float))


class _Question:
    def __init__(self, column, value):
        self.column = column
        self.value  = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value if _is_numeric(val) else val == self.value


class _Leaf:
    def __init__(self, rows):
        counts = {}
        for row in rows:
            lbl = row[-1]
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
    """
    Recursively replace notebook Question/Leaf/Decision_Node classes
    with our local ones so the loaded pickle works correctly.
    """
    if node is None:
        return node

    # It's a Leaf-like object
    if hasattr(node, "predictions") and not hasattr(node, "question"):
        leaf = _Leaf.__new__(_Leaf)
        leaf.predictions = node.predictions
        return leaf

    # It's a Decision_Node-like object
    if hasattr(node, "question"):
        q = node.question
        new_q = _Question.__new__(_Question)
        new_q.column = q.column
        new_q.value  = q.value

        dn = _Decision_Node.__new__(_Decision_Node)
        dn.question     = new_q
        dn.true_branch  = _patch_dt(node.true_branch)
        dn.false_branch = _patch_dt(node.false_branch)
        return dn

    return node


# ---------------------------------------------------------------------------
# Load all artifacts once at import time
# ---------------------------------------------------------------------------

_ARTIFACTS_PATH = "ids_model"

try:
    _gnb_data       = joblib.load(f"{_ARTIFACTS_PATH}/gnb_model.pkl")
    _dt_model_raw   = joblib.load(f"{_ARTIFACTS_PATH}/dt_model.pkl")
    _xgb_model      = joblib.load(f"{_ARTIFACTS_PATH}/xgb_model.pkl")
    _pca_obj        = joblib.load(f"{_ARTIFACTS_PATH}/pca_obj.pkl")
    _numeric_cols   = joblib.load(f"{_ARTIFACTS_PATH}/numeric_cols.pkl")
    _vocab_protocol = joblib.load(f"{_ARTIFACTS_PATH}/vocab_protocol.pkl")
    _vocab_service  = joblib.load(f"{_ARTIFACTS_PATH}/vocab_service.pkl")
    _vocab_flag     = joblib.load(f"{_ARTIFACTS_PATH}/vocab_flag.pkl")

    # Rebuild GNB from plain data (no class pickle issue)
    if isinstance(_gnb_data, dict) and "mean" in _gnb_data:
        _gnb = _GNBPredictor(_gnb_data)
    else:
        # Old-style full object save — wrap it
        _gnb = _GNBPredictor({
            "mean":     _gnb_data.mean,
            "variance": _gnb_data.variance,
            "classes":  _gnb_data.classes,
            "prior":    _gnb_data.prior,
            "n_class":  _gnb_data.n_class,
        })

    # Patch DT so its nodes use our local classes
    _dt_model = _patch_dt(_dt_model_raw)

    _LOADED = True

except Exception as e:
    _LOADED = False
    _LOAD_ERROR = str(e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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


def predict_connection(input_dict: dict):
    """
    Predict whether a network connection is GOOD or BAD.

    Returns
    -------
    result     : str  — "GOOD" or "BAD"
    confidence : int  — % of models that agreed (33, 67, or 100)
    votes      : dict — {"GNB": "GOOD"/"BAD", "DT": ..., "XGB": ...}
    """
    if not _LOADED:
        raise RuntimeError(f"Model artifacts failed to load: {_LOAD_ERROR}")

    input_df = pd.DataFrame([input_dict])

    # ── 1. GNB ───────────────────────────────────────────────────────────
    gnb_numeric = input_df[_numeric_cols].values.astype(float)
    gnb_pca     = _pca_obj.transform(gnb_numeric)
    gnb_vote    = int(_gnb.predict(gnb_pca[0]))

    # ── 2. Decision Tree ─────────────────────────────────────────────────
    dt_input_row = [input_dict.get(col, 0) for col in FEATURE_ORDER]
    dt_input_row.append(0)   # dummy target
    dt_raw  = _classify(dt_input_row, _dt_model)
    dt_vote = int(sorted(dt_raw.items(), key=lambda x: x[1], reverse=True)[0][0])

    # ── 3. XGBoost ───────────────────────────────────────────────────────
    xgb_pca    = _pca_obj.transform(input_df[_numeric_cols].values.astype(float))
    xgb_pca_df = pd.DataFrame(xgb_pca)

    p_vec = CountVectorizer(vocabulary=_vocab_protocol, binary=True).fit_transform(
                [str(input_dict.get('protocol_type', 'tcp'))])
    s_vec = CountVectorizer(vocabulary=_vocab_service, binary=True).fit_transform(
                [str(input_dict.get('service', 'http'))])
    f_vec = CountVectorizer(vocabulary=_vocab_flag, binary=True).fit_transform(
                [str(input_dict.get('flag', 'SF'))])

    xgb_input = pd.DataFrame(hstack((xgb_pca_df, p_vec, s_vec, f_vec)).toarray())
    xgb_vote  = int(_xgb_model.predict(xgb_input)[0])

    # ── 4. Max voting ────────────────────────────────────────────────────
    votes      = [gnb_vote, dt_vote, xgb_vote]
    final_vote = int(np.bincount(votes).argmax())
    confidence = int(votes.count(final_vote) / 3 * 100)
    result     = "GOOD" if final_vote == 1 else "BAD"

    return result, confidence, {
        "GNB": "GOOD" if gnb_vote == 1 else "BAD",
        "DT":  "GOOD" if dt_vote  == 1 else "BAD",
        "XGB": "GOOD" if xgb_vote == 1 else "BAD",
    }
