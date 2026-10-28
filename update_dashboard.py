
from pathlib import Path
import re

path = Path(__file__).with_name("dashboard.py")
source = path.read_text(encoding="utf-8")
original = source

# Keep a backup before modifying the dashboard.
backup = path.with_name("dashboard.py.backup")
if not backup.exists():
    backup.write_text(source, encoding="utf-8")


def replace_once(old, new, description):
    global source
    if old not in source:
        raise ValueError(f"Could not locate the section: {description}")
    source = source.replace(old, new, 1)


# 1. Browser title.
replace_once(
    'page_title="AI-SOC Threat Analyzer",',
    'page_title="AI-SOC Threat Analyzer | ML-Based Intrusion Detection",',
    "browser title",
)

# 2. Rename the top-bar brand.
replace_once(
    "<h1>NetGuard AI</h1>",
    "<h1>AI-SOC Threat Analyzer</h1>",
    "top-bar branding",
)

replace_once(
    "<p>Network Threat Intelligence Platform</p>",
    "<p>ML-Based Intrusion Detection with RAG-Powered Threat Investigation</p>",
    "top-bar subtitle",
)

# 3. Remove the raw-flow-data expander only.
raw_section = re.compile(
    r'\n    if raw_df is not None:\n'
    r'        with st\.expander\("🔬  Raw Flow Data \(for technical users\)"\):'
    r'.*?(?=\n\ndef render_live_capture_tab|\n\ndef main\(\))',
    re.S,
)
source, removed = raw_section.subn("", source, count=1)
if not removed:
    raise ValueError("Could not locate the Raw Flow Data section.")

# 4. Remove the live-capture function.
live_section = re.compile(
    r'\n\ndef render_live_capture_tab\(\):.*?(?=\n\ndef main\(\))',
    re.S,
)
source, removed = live_section.subn("", source, count=1)
if not removed:
    raise ValueError("Could not locate the live-capture function.")

# 5. Replace main() while leaving render_results() and its charts intact.
main_start = source.find("\ndef main():")
main_end = source.find('\n\nif __name__ == "__main__":', main_start)

if main_start == -1 or main_end == -1:
    raise ValueError("Could not locate the main() function boundaries.")

new_main = r'''

def render_about_content():
    st.markdown("## About AI-SOC Threat Analyzer")
    st.markdown("""
    <div class="ng-callout">
      <b>AI-SOC Threat Analyzer</b> analyzes uploaded network capture files
      to identify potentially suspicious network activity using machine learning.

      <br><br><b>Detection workflow</b><br>
      1. Read the uploaded PCAP or PCAPNG file.<br>
      2. Extract network-flow features using NFStream and Scapy.<br>
      3. Run the available trained ML models.<br>
      4. Present the threat verdict, metrics, and visualizations.<br>
      5. Planned extension: investigate detections using RAG and security evidence.

      <br><br><b>Dataset</b><br>
      UNSW-NB15, as documented by the existing project. The model artifacts
      and training configuration should be checked before making claims about
      exact training volume or evaluation performance.

      <br><br><b>Models</b><br>
      Random Forest, LightGBM, XGBoost, and Isolation Forest.
      The existing prediction implementation determines how their outputs
      contribute to the final verdict.
    </div>
    """, unsafe_allow_html=True)


def main():
    # Top-right About button; no live-capture navigation.
    header_col, about_col = st.columns([8, 1])

    with header_col:
        if "upload_result" in st.session_state:
            verdict = st.session_state.upload_result.get("verdict", "NORMAL")
            topbar_status = "attack" if verdict == "ATTACK" else "normal"
        else:
            topbar_status = "ready"
        render_topbar(topbar_status)

    with about_col:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        if st.button("ⓘ About", key="top_about", use_container_width=True):
            st.session_state["show_about_page"] = not st.session_state.get(
                "show_about_page", False
            )

    if st.session_state.get("show_about_page", False):
        if st.button("← Back to analysis", key="back_to_analysis"):
            st.session_state["show_about_page"] = False
            st.rerun()
        render_about_content()
        return

    # PCAP introduction.
    st.markdown("""
    <div class="ng-callout" style="margin-bottom:1rem">
      <b>What is a PCAP file?</b>
      A PCAP file records network packets captured from network communication.
      It is similar to a recording of network activity that can be reviewed later.
      Upload a <b>.pcap</b> or <b>.pcapng</b> file to extract network flows and
      check them for potentially suspicious patterns.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a PCAP or PCAPNG file",
        type=["pcap", "pcapng", "cap"],
        help="Supported formats: .pcap, .pcapng, and .cap",
    )

    # Dataset and model information below the uploader.
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d45;
                border-radius:12px;padding:1rem 1.2rem;margin:0.8rem 0 1.2rem;">
      <div class="ng-panel-title"><span>🧠</span> Detection Technology</div>
      <b style="color:#00d4ff">Dataset</b><br>
      UNSW-NB15<br><br>
      <b style="color:#00d4ff">ML models</b><br>
      Random Forest · LightGBM · XGBoost · Isolation Forest
      <br><br>
      <span style="color:#94a3b8;font-size:0.8rem">
      Network flows are extracted from the uploaded capture and passed to the
      existing prediction pipeline. Model fusion details depend on the current
      implementation in prediction.py.
      </span>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is not None:
        file_size = len(uploaded.getvalue())
        st.caption(f"Selected file: {uploaded.name} · {human_bytes(file_size)}")

        if st.button("🔍 Analyze File", key="analyze_file"):
            pcap_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(uploaded.name).suffix or ".pcap",
                    delete=False,
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    pcap_path = tmp.name

                progress_placeholder = st.empty()
                steps = [
                    ("📡 Parsing packet headers…", 0.15),
                    ("🔬 Extracting flow features…", 0.40),
                    ("🤖 Running ML threat detection…", 0.70),
                    ("🔍 Checking for anomalies…", 0.85),
                    ("📊 Building the analysis report…", 1.00),
                ]

                # Preserve the existing analysis pipeline and progress UI.
                with progress_placeholder.container():
                    bar = st.progress(0)
                    msg = st.empty()
                    status = st.empty()

                    with status:
                        with st.spinner("Analyzing uploaded network traffic…"):
                            result, raw_df, err = run_full_analysis(pcap_path)

                    if err is None:
                        for label, progress in steps:
                            msg.markdown(
                                f'<div class="ng-callout">{label}</div>',
                                unsafe_allow_html=True,
                            )
                            bar.progress(progress)

                    if err:
                        st.error(f"Analysis error: {err}")
                    else:
                        st.session_state.upload_result = result
                        st.session_state.upload_raw_df = raw_df
                        st.session_state.upload_name = uploaded.name
                        bar.progress(1.0)
                        msg.success("Analysis completed.")
                        st.rerun()

            except Exception as exc:
                st.error(f"Could not analyze the uploaded file: {exc}")
            finally:
                if pcap_path and Path(pcap_path).exists():
                    try:
                        Path(pcap_path).unlink()
                    except OSError:
                        pass

    if "upload_result" in st.session_state:
        render_results(
            st.session_state.upload_result,
            st.session_state.upload_raw_df,
            filename=st.session_state.get("upload_name", "capture.pcap"),
        )


if __name__ == "__main__":
    main()
'''

source = source[:main_start] + new_main

# Sanity checks: ensure chart and terminology functions remain present.
required = [
    "def render_results(",
    "Per-Flow Threat Score",
    "Traffic Breakdown",
    "Protocol Distribution",
    "Top Services Contacted",
    "Threat Score Distribution",
    "Data Volume per Connection",
    "What do these terms mean?",
    "AI-SOC Threat Analyzer",
]
missing = [item for item in required if item not in source]
if missing:
    raise ValueError(f"Safety check failed; missing: {missing}")

if "render_live_capture_tab" in source:
    raise ValueError("Safety check failed: live-capture function still exists.")

if "Raw Flow Data (for technical users)" in source:
    raise ValueError("Safety check failed: raw-flow section still exists.")

path.write_text(source, encoding="utf-8")
print(f"Updated {path}")
print(f"Backup saved to {backup}")
print("Existing render_results() chart code was retained.")
