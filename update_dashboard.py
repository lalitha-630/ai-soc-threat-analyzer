import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI-SOC Threat Analyzer",
    page_icon="🛡️",
    layout="wide",
)

MODELS_DIR = Path(__file__).resolve().parent / "models"

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp { background: #0b0f1a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #111827; }
    .block-container { padding-top: 1.5rem; }
    .soc-header {
        padding: 1.2rem 1.5rem; margin-bottom: 1rem;
        background: linear-gradient(120deg, #111827, #17243a);
        border: 1px solid #26364f; border-radius: 12px;
    }
    .soc-header h1 { margin: 0; color: #00d4ff; }
    .soc-header p { margin: .4rem 0 0; color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model loading and analysis ----------
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    from prediction import load_artifacts
    return load_artifacts(str(MODELS_DIR))


def analyze_pcap(pcap_path):
    from feature_extraction import extract_all_features
    from prediction import run_prediction

    raw_df = extract_all_features(pcap_path)

    if raw_df is None or raw_df.empty:
        raise ValueError("No network flows were extracted from this PCAP.")

    artifacts = load_model_artifacts()
    result = run_prediction(raw_df, artifacts)

    # Confirm that each returned score/prediction corresponds to one flow.
    expected_rows = len(raw_df)
    required_arrays = (
        "all_flow_scores",
        "ensemble_scores",
        "anomaly_scores",
        "xgb_scores",
        "lgbm_scores",
        "rf_scores",
        "flow_predictions",
        "isolation_predictions",
    )

    for key in required_arrays:
        if key not in result:
            raise KeyError(f"prediction.py did not return required key: {key}")
        if len(result[key]) != expected_rows:
            raise ValueError(
                f"{key} has {len(result[key])} values, but the extracted "
                f"flow data has {expected_rows} rows."
            )

    return result, raw_df


# ---------- Display helpers ----------
def score_percent(value):
    return f"{float(value) * 100:.1f}%"


def make_flow_table(result, raw_df):
    """Build a row-aligned table: row i always describes raw_df.iloc[i]."""
    flow_table = raw_df.reset_index(drop=True).copy()

    flow_table.insert(0, "Flow", np.arange(1, len(flow_table) + 1))
    flow_table["XGBoost score"] = result["xgb_scores"]
    flow_table["LightGBM score"] = result["lgbm_scores"]
    flow_table["Random Forest score"] = result["rf_scores"]
    flow_table["Ensemble score"] = result["ensemble_scores"]
    flow_table["Isolation Forest anomaly score"] = result["anomaly_scores"]
    flow_table["Combined score"] = result["all_flow_scores"]
    flow_table["Isolation Forest prediction"] = result["isolation_predictions"]
    flow_table["Final prediction"] = np.where(
        np.asarray(result["flow_predictions"]) == 1,
        "Suspicious",
        "Not flagged",
    )
    return flow_table


def render_results(result, raw_df, filename):
    scores = np.asarray(result["all_flow_scores"], dtype=float)
    predictions = np.asarray(result["flow_predictions"], dtype=int)
    threshold = float(result["detection_threshold"])
    total_flows = len(raw_df)
    flagged_flows = int(np.sum(predictions == 1))
    normal_flows = total_flows - flagged_flows
    attack_ratio = flagged_flows / total_flows if total_flows else 0

    threat_detected = flagged_flows > 0
    verdict_label = "THREAT DETECTED" if threat_detected else "NO THREAT DETECTED"
    verdict_color = "#ef4444" if threat_detected else "#10b981"

    st.markdown(
        f"""
        <div class="soc-header">
            <h1>AI-SOC Threat Analyzer</h1>
            <p>PCAP analysis · {filename}</p>
            <h2 style="color:{verdict_color}; margin-bottom:0">{verdict_label}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "The combined score is a weighted score, not a calibrated probability. "
        "The threshold below is the threshold configured in prediction.py."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total flows", f"{total_flows:,}")
    m2.metric("Flagged flows", f"{flagged_flows:,}")
    m3.metric("Flagged share", f"{attack_ratio * 100:.1f}%")
    m4.metric("Detection threshold", f"{threshold:.4f}")

    st.subheader("Combined score by flow")
    chart_df = pd.DataFrame({
        "Flow": np.arange(1, total_flows + 1),
        "Combined score": scores,
        "Prediction": np.where(predictions == 1, "Suspicious", "Not flagged"),
    })
    fig = go.Figure()
    for label, color in [("Not flagged", "#10b981"), ("Suspicious", "#ef4444")]:
        part = chart_df[chart_df["Prediction"] == label]
        fig.add_trace(go.Bar(
            x=part["Flow"],
            y=part["Combined score"],
            name=label,
            marker_color=color,
            hovertemplate="Flow %{x}<br>Combined score: %{y:.4f}<extra></extra>",
        ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Configured threshold: {threshold:.4f}",
    )
    fig.update_layout(
        barmode="overlay",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="Flow number",
        yaxis_title="Combined score",
        legend_title="Final prediction",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flow-level detection results")
    flow_table = make_flow_table(result, raw_df)

    flagged_only = st.checkbox("Show only flagged flows", value=False)
    displayed_table = (
        flow_table[flow_table["Final prediction"] == "Suspicious"]
        if flagged_only
        else flow_table
    )

    st.dataframe(displayed_table, use_container_width=True, hide_index=True)

    csv = displayed_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download displayed flow results as CSV",
        data=csv,
        file_name="ai_soc_flow_results.csv",
        mime="text/csv",
    )

    st.subheader("Model score comparison")
    model_score_df = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "Random Forest", "Ensemble", "Isolation Forest"],
        "Mean score": [
            np.mean(result["xgb_scores"]),
            np.mean(result["lgbm_scores"]),
            np.mean(result["rf_scores"]),
            np.mean(result["ensemble_scores"]),
            np.mean(result["anomaly_scores"]),
        ],
    })
    score_fig = go.Figure(go.Bar(
        x=model_score_df["Model"],
        y=model_score_df["Mean score"],
        marker_color=["#7c3aed", "#00d4ff", "#10b981", "#f59e0b", "#ef4444"],
        hovertemplate="%{x}<br>Mean score: %{y:.4f}<extra></extra>",
    ))
    score_fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis_title="Mean score",
    )
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Traffic overview")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Final flow predictions**")
        pie = go.Figure(go.Pie(
            labels=["Not flagged", "Suspicious"],
            values=[normal_flows, flagged_flows],
            hole=0.55,
            marker_colors=["#10b981", "#ef4444"],
        ))
        pie.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(pie, use_container_width=True)

    with c2:
        st.markdown("**Protocol distribution**")
        if "proto" in raw_df.columns:
            proto_counts = raw_df["proto"].astype(str).value_counts()
            proto_fig = go.Figure(go.Bar(
                x=proto_counts.index,
                y=proto_counts.values,
                marker_color="#00d4ff",
            ))
            proto_fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis_title="Protocol",
                yaxis_title="Flows",
            )
            st.plotly_chart(proto_fig, use_container_width=True)
        else:
            st.info("The extracted flow data has no `proto` column.")

    st.subheader("Detection summary")
    st.write(f"- Final verdict: **{result['verdict']}**")
    st.write(f"- Flows flagged by the final decision: **{result['flagged_flows']}**")
    st.write(f"- Flows analyzed: **{result['total_flows']}**")
    st.write(f"- Maximum combined score: **{float(result['confidence']):.4f}**")
    st.write(
        "- Isolation Forest's score is a relative anomaly score, "
        "not an attack probability."
    )

    with st.expander("What the scores mean"):
        st.markdown(
            """
            - **XGBoost, LightGBM, Random Forest:** model scores for class 1.
            - **Ensemble score:** the mean of the three supervised model scores.
            - **Isolation Forest anomaly score:** a relative anomaly signal scaled within
              this PCAP's extracted flows; it is not a calibrated attack probability.
            - **Combined score:** the weighted combination used by `prediction.py`.
            - **Final prediction:** suspicious when the combined score meets or exceeds
              the configured detection threshold.
            - **Flow number:** row position in the extracted flow data. Flow *n* in this
              table corresponds to `raw_df.iloc[n - 1]`.
            """
        )


# ---------- Main app ----------
def main():
    with st.sidebar:
        st.title("AI-SOC")
        page = st.radio("Navigation", ["Upload & Analyze", "About"])
        st.divider()
        st.caption("Models: XGBoost · LightGBM · Random Forest · Isolation Forest")
        st.caption("Feature extraction: NFStream / Scapy")

    if page == "About":
        st.title("About AI-SOC Threat Analyzer")
        st.write(
            "Upload a PCAP file to extract network flows and evaluate them with "
            "the trained intrusion-detection models."
        )
        st.write(
            "The dashboard shows the model scores and final decision for each "
            "extracted flow. RAG investigation is not performed by this dashboard "
            "unless you connect it to a separate investigation component."
        )
        return

    st.title("AI-SOC Threat Analyzer")
    st.write("Upload a PCAP/PCAPNG capture for flow-level ML threat analysis.")

    if not MODELS_DIR.is_dir():
        st.error(
            f"Models directory not found: `{MODELS_DIR}`. "
            "Place the trained model and preprocessing artifacts in that directory."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload PCAP file",
        type=["pcap", "pcapng", "cap"],
    )

    if uploaded_file is None:
        st.info("Choose a PCAP file to begin.")
        return

    st.caption(
        f"Selected file: {uploaded_file.name} · "
        f"{len(uploaded_file.getvalue()):,} bytes"
    )

    if st.button("Analyze PCAP", type="primary"):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix or ".pcap",
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner("Extracting flows and running the trained models..."):
                result, raw_df = analyze_pcap(temp_path)

            st.session_state["analysis_result"] = result
            st.session_state["analysis_raw_df"] = raw_df
            st.session_state["analysis_filename"] = uploaded_file.name

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if "analysis_result" in st.session_state:
        render_results(
            st.session_state["analysis_result"],
            st.session_state["analysis_raw_df"],
            st.session_state.get("analysis_filename", "uploaded.pcap"),
        )


if __name__ == "__main__":
    main()
