import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

# =========================================================
# DUMMY CLASSES (FIX FOR PICKLE ERROR)
# These must be injected into __main__ BEFORE joblib.load()
# so that pickle can resolve them regardless of which file
# is the entry point (app.py, predictor.py, etc.)
# =========================================================

class Decision_Node:
    pass

class Leaf:
    pass

class Question:
    pass

# ── Inject into __main__ so pickle finds them there too ──
import __main__
for _cls in (Decision_Node, Leaf, Question):
    if not hasattr(__main__, _cls.__name__):
        setattr(__main__, _cls.__name__, _cls)

# =========================================================
# GNB MODEL
# =========================================================

class _GNBPredictor:
    def __init__(self, data: dict):
        self.mean     = data["mean"]
        self.variance = data["variance"]
        self.classes  = data["classes"]
        self.prior    = data["prior"]
        self.n_class  = data["n_class"]

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

        for cls_mv in split:
            prob = np.prod([
                self._gnb_base(X[j], cls_mv[j][0], cls_mv[j][1])
                for j in range(len(cls_mv))
            ]) * self.prior
            final_probs.append(prob)

        return self.classes[final_probs.index(max(final_probs))]

# =========================================================
# DECISION TREE HELPERS
# =========================================================

def _is_numeric(value):
    return isinstance(value, (int, float))

class _Question:
    def __init__(self, column, value):
        self.column = column
        self.value  = value

    def match(self, example):
        val = example[self.column]
        if _is_numeric(val):
            return val >= self.value
        return val == self.value

class _Leaf:
    def __init__(self, rows):
        counts = {}
        for row in rows:
            lbl          = row[-1]
            counts[lbl]  = counts.get(lbl, 0) + 1
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

    # Leaf node
    if hasattr(node, "predictions") and not hasattr(node, "question"):
        leaf             = _Leaf.__new__(_Leaf)
        leaf.predictions = node.predictions
        return leaf

    # Decision node
    if hasattr(node, "question"):
        q        = node.question
        new_q    = _Question.__new__(_Question)
        new_q.column = q.column
        new_q.value  = q.value

        dn             = _Decision_Node.__new__(_Decision_Node)
        dn.question    = new_q
        dn.true_branch  = _patch_dt(node.true_branch)
        dn.false_branch = _patch_dt(node.false_branch)
        return dn

    return node

# =========================================================
# LOAD MODELS
# =========================================================

_ARTIFACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ids_model")

try:
    _gnb_data      = joblib.load(os.path.join(_ARTIFACTS_PATH, "gnb_model.pkl"))
    _dt_model_raw  = joblib.load(os.path.join(_ARTIFACTS_PATH, "dt_model.pkl"))
    _xgb_model     = joblib.load(os.path.join(_ARTIFACTS_PATH, "xgb_model.pkl"))
    _pca_obj       = joblib.load(os.path.join(_ARTIFACTS_PATH, "pca_obj.pkl"))
    _numeric_cols  = joblib.load(os.path.join(_ARTIFACTS_PATH, "numeric_cols.pkl"))
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
    # Only pass numeric feature columns + dummy label; never raw dict.values()
    dt_input_row = [input_dict[col] for col in _numeric_cols]
    dt_input_row.append(0)          # dummy label required by _classify
    dt_raw   = _classify(dt_input_row, _dt_model)
    dt_vote  = int(max(dt_raw, key=dt_raw.get))

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
    # Align column names to avoid XGBoost feature-name mismatch
    xgb_input.columns = [str(i) for i in range(xgb_input.shape[1])]

    xgb_vote = int(_xgb_model.predict(xgb_input)[0])

    # --- Majority Voting ---
    votes      = [gnb_vote, dt_vote, xgb_vote]
    final_vote = int(np.bincount(votes).argmax())
    confidence = int(votes.count(final_vote) / 3 * 100)

    result = "GOOD" if final_vote == 1 else "BAD"

    return result, confidence, {
        "GNB": "GOOD" if gnb_vote == 1 else "BAD",
        "DT":  "GOOD" if dt_vote  == 1 else "BAD",
        "XGB": "GOOD" if xgb_vote == 1 else "BAD",
    }
