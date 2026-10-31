import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Feature configuration
# ============================================================

ALL_42_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
    "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]

TO_FILL_ZERO = [
    "trans_depth",
    "response_body_len",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
]

CAT_COLS_FALLBACK = {
    "proto": "unknown",
    "service": "-",
    "state": "unknown",
}


# ============================================================
# Artifact loading
# ============================================================

def load_artifacts(artifacts_dir="models"):
    """
    Load trained models and preprocessing artifacts.

    Expected artifact files:
      - xgb_model.pkl
      - lgbm_model.pkl
      - rf_model.pkl
      - isolation_forest.pkl
      - scaler.pkl
      - encoders.pkl
      - threshold.pkl

    Update these filenames if your project uses different names.
    """
    artifacts_dir = Path(artifacts_dir)

    def load_pickle(filename):
        path = artifacts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact not found: {path}")
        return joblib.load(path)

    artifacts = {
        "xgb_model": load_pickle("xgb_model.pkl"),
        "lgbm_model": load_pickle("lgbm_model.pkl"),
        "rf_model": load_pickle("rf_model.pkl"),
        "isolation_forest": load_pickle("isolation_forest.pkl"),
        "scaler": load_pickle("scaler.pkl"),
        "encoders": load_pickle("encoders.pkl"),
        "threshold": load_pickle("threshold.pkl"),
    }

    return artifacts


# ============================================================
# Preprocessing
# ============================================================

def preprocess(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Convert extracted raw flow data into the model's expected features.

    Keeps the input row order so prediction index i corresponds to
    raw_df.iloc[i] in the dashboard.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("No network flows were extracted from the PCAP.")

    df = raw_df.copy()

    # Add any expected feature missing from the extracted flow data.
    for col in ALL_42_FEATURES:
        if col not in df.columns:
            if col in CAT_COLS_FALLBACK:
                df[col] = CAT_COLS_FALLBACK[col]
            else:
                df[col] = 0

    # Fill missing values in selected features.
    for col in TO_FILL_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Encode categorical features.
    encoders = artifacts.get("encoders", {})
    for col, encoder in encoders.items():
        if col not in df.columns:
            continue

        values = df[col].fillna(CAT_COLS_FALLBACK.get(col, "unknown")).astype(str)

        # Handle values not seen during training.
        known_values = set(getattr(encoder, "classes_", []))
        if known_values:
            fallback = CAT_COLS_FALLBACK.get(col, next(iter(known_values)))
            values = values.map(lambda value: value if value in known_values else fallback)

        df[col] = encoder.transform(values)

    # Select features in the exact expected order.
    X_df = df[ALL_42_FEATURES].copy()

    # Ensure model input is numeric.
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = artifacts.get("scaler")
    if scaler is not None:
        X_scaled = scaler.transform(X_df)
        X_df = pd.DataFrame(
            X_scaled,
            columns=ALL_42_FEATURES,
            index=X_df.index,
        )

    return X_df


# ============================================================
# Model score helpers
# ============================================================

def _positive_class_scores(model, X_df) -> np.ndarray:
    """
    Return the probability for class 1 when predict_proba is available.
    Otherwise, convert decision_function scores to a 0..1 range.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_df)

        classes = getattr(model, "classes_", None)
        if classes is not None:
            positive_indices = np.where(np.asarray(classes) == 1)[0]
            positive_index = int(positive_indices[0]) if len(positive_indices) else -1
        else:
            positive_index = -1

        return np.asarray(probabilities[:, positive_index], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_df), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    raise TypeError(
        f"{type(model).__name__} does not support predict_proba "
        "or decision_function."
    )


# ============================================================
# Supervised ensemble
# ============================================================

def run_ensemble(X_df: pd.DataFrame, artifacts: dict) -> dict:
    """
    Run XGBoost, LightGBM, and Random Forest.

    Returns each model's per-flow score and the mean ensemble score.
    """
    xgb_scores = _positive_class_scores(artifacts["xgb_model"], X_df)
    lgbm_scores = _positive_class_scores(artifacts["lgbm_model"], X_df)
    rf_scores = _positive_class_scores(artifacts["rf_model"], X_df)

    ensemble_prob = np.mean(
        np.vstack([xgb_scores, lgbm_scores, rf_scores]),
        axis=0,
    )

    return {
        "xgb_scores": xgb_scores,
        "lgbm_scores": lgbm_scores,
        "rf_scores": rf_scores,
        "ensemble_prob": ensemble_prob,
    }


# ============================================================
# Isolation Forest
# ============================================================

def run_isolation_forest(X_df: pd.DataFrame, artifacts: dict):
    """
    Run Isolation Forest and convert its anomaly scores to 0..1.

    The returned anomaly scores are relative anomaly scores, not
    calibrated attack probabilities.
    """
    model = artifacts["isolation_forest"]

    # Isolation Forest: lower decision_function values are more anomalous.
    raw_scores = np.asarray(model.decision_function(X_df), dtype=float)
    anomaly_scores = -raw_scores

    score_min = float(np.min(anomaly_scores))
    score_max = float(np.max(anomaly_scores))

    if np.isclose(score_min, score_max):
        normalized_scores = np.zeros_like(anomaly_scores, dtype=float)
    else:
        normalized_scores = (
            (anomaly_scores - score_min) / (score_max - score_min)
        )

    # sklearn Isolation Forest uses -1 for anomaly and 1 for normal.
    raw_predictions = np.asarray(model.predict(X_df))
    anomaly_predictions = (raw_predictions == -1).astype(int)

    return normalized_scores, anomaly_predictions


# ============================================================
# Combined decision
# ============================================================

def combine_and_decide(
    ensemble_prob: np.ndarray,
    if_attack_prob: np.ndarray,
    artifacts: dict,
) -> dict:
    """
    Combine supervised ensemble scores and Isolation Forest scores.

    The combined score is a weighted score, not a Bayesian probability.
    """
    ensemble_prob = np.asarray(ensemble_prob, dtype=float)
    if_attack_prob = np.asarray(if_attack_prob, dtype=float)

    if len(ensemble_prob) != len(if_attack_prob):
        raise ValueError("Ensemble and Isolation Forest score counts do not match.")

    if len(ensemble_prob) == 0:
        raise ValueError("No flow scores were produced.")

    ensemble_weight = float(artifacts.get("ensemble_weight", 0.5))
    if_weight = float(artifacts.get("if_weight", 0.5))

    total_weight = ensemble_weight + if_weight
    if total_weight <= 0:
        raise ValueError("The model weights must sum to a positive value.")

    ensemble_weight /= total_weight
    if_weight /= total_weight

    threshold = float(np.asarray(artifacts["threshold"]).squeeze())

    all_flow_scores = (
        ensemble_weight * ensemble_prob
        + if_weight * if_attack_prob
    )

    final_predictions = (all_flow_scores >= threshold).astype(int)
    attack_count = int(np.sum(final_predictions == 1))
    total_flows = int(len(final_predictions))
    attack_ratio = attack_count / total_flows

    if attack_count > 0:
        verdict = "Threat Detected"
    else:
        verdict = "No Threat Detected"

    return {
        "verdict": verdict,
        "all_flow_scores": all_flow_scores,
        "ensemble_scores": ensemble_prob,
        "anomaly_scores": if_attack_prob,
        "flow_predictions": final_predictions,
        "flagged_flows": attack_count,
        "total_flows": total_flows,
        "attack_ratio": attack_ratio,
        "detection_threshold": threshold,
        "confidence": float(np.max(all_flow_scores)),
    }


# ============================================================
# Full prediction pipeline
# ============================================================

def run_prediction(raw_df: pd.DataFrame, artifacts: dict) -> dict:
    """
    Run preprocessing, all four models, and the combined decision.

    The returned arrays preserve raw_df row order. Therefore,
    result["flow_predictions"][i] and the score arrays at index i
    correspond to raw_df.iloc[i].
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("No network flows were extracted from the PCAP.")

    X_df = preprocess(raw_df, artifacts)

    if_attack_prob, isolation_predictions = run_isolation_forest(
        X_df,
        artifacts,
    )

    ensemble_results = run_ensemble(X_df, artifacts)

    result = combine_and_decide(
        ensemble_results["ensemble_prob"],
        if_attack_prob,
        artifacts,
    )

    result["xgb_scores"] = ensemble_results["xgb_scores"]
    result["lgbm_scores"] = ensemble_results["lgbm_scores"]
    result["rf_scores"] = ensemble_results["rf_scores"]
    result["isolation_predictions"] = isolation_predictions

    return result
