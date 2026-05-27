🛡️ NetGuard AI — Network Threat Intelligence Platform

A Machine Learning–based Network Intrusion Detection System (NIDS) that analyzes PCAP files and live network traffic to detect malicious activity using an ensemble of classifiers combined with anomaly detection via Bayesian fusion.

Live Demo: netguard-nids.streamlit.app  |  GitHub: 27Aditi/netguard-nids

📌 Table of Contents

About the Project
Technology Stack
Features
ML Pipeline
Project Structure
Installation & Setup
Screenshots
Dataset
Team Members


About the Project
NetGuard AI is a full-stack network intrusion detection system built entirely with Streamlit. It processes .pcap / .pcapng capture files and classifies network flows as NORMAL or ATTACK. It extracts 32 flow-level features using NFStream and Scapy, runs them through a trained ensemble (Random Forest + LightGBM + XGBoost) and an Isolation Forest anomaly detector, then combines the outputs via Bayesian fusion to produce a final threat verdict with risk level, threat intensity, and analyst notes.
The system also supports live capture mode on Linux/macOS (using Scapy's AsyncSniffer) and provides a real-time threat score chart, session history, and model info panel — all within the Streamlit dashboard.

Technology Stack
LayerTechnologyDashboard / UIStreamlitCharts & VisualizationPlotlyPacket ProcessingNFStream, ScapyML — SupervisedRandom Forest, XGBoost, LightGBM (scikit-learn)ML — UnsupervisedIsolation ForestFusion StrategyBayesian Combination (50/50 weighted)DeploymentStreamlit CloudLanguagePython 3.10+

Features

📁 PCAP File Analysis — Upload any .pcap or .pcapng file for instant threat analysis
🎙️ Live Network Capture — Record live traffic from a network interface using Scapy's AsyncSniffer and analyze on-the-fly (Linux/macOS; requires root/admin privileges)
🤖 Ensemble Classification — Random Forest, LightGBM, and XGBoost vote together for robust supervised predictions
🔍 Anomaly Detection — Isolation Forest detects zero-day / unseen attack patterns without labels
⚖️ Bayesian Fusion — 50/50 weighted combination of ensemble probability and anomaly score for a single unified threat score
📊 Real-Time Threat Chart — Rolling average line chart of per-flow Bayesian scores powered by Plotly
📋 Human-Readable Verdict — Risk Level (Low / Medium / High), Network Status (Safe / Suspicious / Danger), Threat Intensity %, Traffic Behavior (Stable / Unstable / Erratic), and Analyst Note
🕓 Session History — All past analysis sessions stored and accessible in the sidebar via Streamlit session state
32 Flow-Level Features — Carefully engineered from the UNSW-NB15 feature set
9 Attack Categories Covered — Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms


ML Pipeline
PCAP File
    │
    ▼
NFStream  ──────────────────────► Flow-level DataFrame (bidirectional stats)
    │
    ▼
Scapy (single-pass)  ───────────► Raw packet features
                                   (TTL, TCP window, RTT, HTTP depth,
                                    FTP login, response body length)
    │
    ▼
32 Features Extracted
    │
    ├── Label Encoding  (proto, service, state)
    └── MinMax Scaling
    │
    ▼
┌─────────────────────────────────────────────┐
│  Ensemble Classifier                        │
│  RF + LightGBM + XGBoost → avg probability │
└──────────────────────┬──────────────────────┘
                       │           ┌──────────────────────────┐
                       │           │  Isolation Forest        │
                       │           │  anomaly score → [0, 1]  │
                       │           └────────────┬─────────────┘
                       ▼                        ▼
              Bayesian Combination  (0.5 × ensemble + 0.5 × IF)
                       │
                       ▼
             Combined Score ≥ Threshold?
                       │
              ┌────────┴─────────┐
           ATTACK             NORMAL
Features Extracted (32)
dur · proto · service · state · spkts · dpkts · sbytes · dbytes · sttl · dttl · sload · dload · sloss · dloss · sinpkt · swin · dwin · tcprtt · synack · smean · dmean · trans_depth · response_body_len · ct_srv_src · ct_state_ttl · ct_src_dport_ltm · ct_dst_sport_ltm · ct_dst_src_ltm · is_ftp_login · ct_flw_http_mthd · ct_srv_dst · is_sm_ips_ports

Project Structure
netguard-nids/
│
├── dashboard.py                         ← Streamlit app (all pages & UI)
│
├── pipelines/
│   ├── feature_extraction.py            ← NFStream + Scapy feature extraction
│   └── prediction.py                    ← Prediction pipeline + verdict logic
│
├── models/
│   ├── classifiers/
│   │   ├── final_rf.pkl                 ← Random Forest
│   │   ├── final_lgbm.pkl               ← LightGBM
│   │   └── final_xgb.pkl                ← XGBoost
│   ├── anomaly/
│   │   └── isolation_forest.pkl
│   ├── scalers/
│   │   ├── nidss_scaler.pkl             ← Ensemble scaler
│   │   └── if_scaler.pkl                ← IF scaler
│   ├── encoders/
│   │   └── nidss_encoders.pkl           ← Label encoders (proto, service, state)
│   ├── feature_order/
│   │   ├── nidss_features.pkl
│   │   └── if_feature_order.pkl
│   └── threshold/
│       └── final_threshold.pkl
│
├── requirements.txt
└── README.md

Installation & Setup
Prerequisites

Python 3.10+
Npcap (Windows) or libpcap (Linux/macOS) — required by Scapy and NFStream

1. Clone the Repository
bashgit clone https://github.com/27Aditi/netguard-nids.git
cd netguard-nids
2. Create and Activate a Virtual Environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
3. Install Dependencies
bashpip install -r requirements.txt
Or manually:
bashpip install streamlit plotly nfstream scapy
pip install xgboost lightgbm scikit-learn joblib
4. Run the App
bashstreamlit run dashboard.py
Open http://localhost:8501 in your browser.

Live Capture requires root/admin privileges. On Linux/macOS run sudo streamlit run dashboard.py. On Windows, use Wireshark to capture a .pcap file and upload it via the Upload & Analyze tab instead.

5. Add Your Trained Models
Place your .pkl model files in the models/ directory following the structure above. If no models are found, the app runs in demo mode with simulated results.

Screenshots
Live Capture Mode
Show Image

The Live Network Capture tab lets users start real-time packet recording directly from a network interface. Captured traffic is automatically saved and analyzed when recording stops.

Threat Detected — Upload & Analyze Mode
Show Image

After uploading a PCAP file and clicking Analyze, the dashboard displays the full verdict including threat status, risk level, and the AI-generated analyst note.


Dataset
Trained on the UNSW-NB15 dataset — a comprehensive, modern network intrusion dataset generated by the IXIA PerfectStorm tool at the Australian Centre for Cyber Security (ACCS).
PropertyValueTotal Flows2.5 million+Attack Categories9 (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)Features Used32 (subset of UNSW-NB15 feature set)

Team Members
NameAditi BhatnagarAbhinay Jat
B.Tech — Computer Science & Engineering

License
This project is intended for educational and research purposes.
