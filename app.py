"""
MEC-AI HetFL: Interactive Prototype & Dashboard
Adaptive Federated Learning for Resource-Constrained IoT Devices
Birla Institute of Technology, Mesra · Patna Campus
Reference: Mughal et al., Scientific Reports (Nature Portfolio), 2024
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys

# Ensure current directory is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.dataset import IoTDatasetGenerator, DOMAIN_PROFILES
from core.device import ConstrainedIoTDevice
from core.edge_ai_node import EdgeAINode
from core.hetero_server import HeteroFederatedServer
from core.baselines import run_synchronous_fedavg, get_paper_benchmark_table

# Page Configuration
st.set_page_config(
    page_title="MEC-AI HetFL Prototype",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 20px;">
    <h2 style="margin: 0; padding-bottom: 6px; color: #ffffff;">⚡ MEC-AI HetFL: Interactive Federated Learning Prototype</h2>
    <p style="margin: 0; font-size: 15px; opacity: 0.9;">
        <strong>Adaptive Federated Learning for Resource-Constrained IoT Devices through Edge Intelligence and Multi-Edge Clustering</strong><br>
        <em>Department of Computer Science & Engineering · B.Tech Project-1</em>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Hyperparameters")

with st.sidebar.expander("🌐 Topology & Cluster Settings", expanded=True):
    devices_per_cluster = st.slider("IoT Devices per Cluster", min_value=3, max_value=12, value=6, step=1)
    straggler_rate = st.slider("Straggler Injection Rate (%)", min_value=10, max_value=60, value=35, step=5) / 100.0

with st.sidebar.expander("🎯 Algorithm 2: Node Selection Weights", expanded=True):
    st.markdown("**Dual-Factor Quality Score Formula:**")
    st.latex(r"Q(e) = w_1 \cdot \text{NetworkScore}(e) + w_2 \cdot \text{ExpertiseScore}(e)")
    w1 = st.slider("Weight $w_1$ (Network Quality)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    w2 = round(1.0 - w1, 2)
    st.info(f"Weight $w_2$ (AI Expertise) automatically balanced to: **{w2}**")
    quality_thresh = st.slider("Quality Threshold ($Q_{min}$ to accept)", min_value=0.2, max_value=0.8, value=0.45, step=0.05)
    selection_ratio = st.slider("Selection Ratio (Top % candidates)", min_value=0.4, max_value=1.0, value=0.75, step=0.05)

with st.sidebar.expander("🔁 Federated Training Parameters", expanded=False):
    num_rounds = st.slider("Federated Rounds", min_value=3, max_value=25, value=10, step=1)
    local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=8, value=3, step=1)
    learning_rate = st.select_slider("Learning Rate (η)", options=[0.005, 0.01, 0.02, 0.03, 0.05], value=0.03)


# Tabs Definition
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Simulation",
    "📱 Multi-Tier Device Inspector",
    "⚖️ Comparative Baselines",
    "📋 Paper Table 1 Benchmarking"
])

# Initialize or Retrieve Simulation Environment
def init_environment():
    gen = IoTDatasetGenerator(random_seed=42)
    domains = ["Smart City", "Mobile Healthcare", "Agriculture"]
    clusters = []
    all_devices = []
    all_test_X, all_test_y = [], []

    for i, domain in enumerate(domains):
        X, y = gen.generate_domain_data(domain, total_samples=1000)
        split_idx = int(0.8 * len(y))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        all_test_X.append(X_test)
        all_test_y.append(y_test)

        partitions = gen.partition_non_iid(X_train, y_train, devices_per_cluster, non_iid_alpha=0.6)
        cluster_node = EdgeAINode(f"Cluster-{i+1}", domain, w1=w1, w2=w2, seed=42 + i)
        cluster_node.set_cluster_data(X_test[:50], y_test[:50])

        rng = np.random.RandomState(42 + i * 15)
        for d_idx, (d_X, d_y) in enumerate(partitions):
            is_straggler = (rng.rand() < straggler_rate)
            dev = ConstrainedIoTDevice(
                device_id=f"Node-{i+1}-{d_idx+1}",
                cluster_id=cluster_node.cluster_id,
                X_train=d_X,
                y_train=d_y,
                is_straggler=is_straggler,
                seed=42 + d_idx
            )
            cluster_node.register_device(dev)
            all_devices.append(dev)

        clusters.append(cluster_node)

    global_X_test = np.concatenate(all_test_X, axis=0)
    global_y_test = np.concatenate(all_test_y, axis=0)

    server = HeteroFederatedServer(clusters=clusters, seed=42)
    server.set_global_test_data(global_X_test, global_y_test)
    server.broadcast_global_model()

    return server, clusters, all_devices, (global_X_test, global_y_test)


# --- TAB 1: LIVE SIMULATION ---
with tab1:
    st.subheader("Interactive Federated Learning Convergence")
    st.markdown("Click the button below to train **MEC-AI HetFL** in real-time across the 3 Edge Clusters.")

    col_btn1, col_btn2 = st.columns([1, 4])
    run_sim = col_btn1.button("▶ Run Live Training", type="primary", use_container_width=True)

    metrics_container = st.empty()
    chart_container = st.empty()
    log_container = st.empty()

    if run_sim:
        server, clusters, all_devices, test_data = init_environment()
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        hetfl_records = []
        chart_data = {
            'Round': [],
            'MEC-AI HetFL Accuracy (%)': [],
            'MEC-AI HetFL Loss': [],
            'Latency (ms)': []
        }

        for r in range(1, num_rounds + 1):
            status_text.text(f"Executing Federated Round {r}/{num_rounds} across 3 Edge AI Clusters...")
            rec = server.run_federated_round(
                round_idx=r,
                selection_ratio=selection_ratio,
                quality_threshold=quality_thresh,
                epochs=local_epochs,
                lr=learning_rate
            )
            hetfl_records.append(rec)
            progress_bar.progress(int(r / num_rounds * 100))

            chart_data['Round'].append(r)
            chart_data['MEC-AI HetFL Accuracy (%)'].append(rec['accuracy'] * 100)
            chart_data['MEC-AI HetFL Loss'].append(rec['loss'])
            chart_data['Latency (ms)'].append(rec['round_latency_ms'])

            # Live Metrics Display
            with metrics_container.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Global Accuracy", f"{rec['accuracy']*100:.2f}%", delta=f"{'+' if r > 1 else ''}{round(rec['accuracy']*100 - (hetfl_records[-2]['accuracy']*100 if r > 1 else 0), 2)}%")
                c2.metric("Cross-Entropy Loss", f"{rec['loss']:.4f}", delta=f"{'-' if r > 1 else ''}{round(abs(rec['loss'] - (hetfl_records[-2]['loss'] if r > 1 else 0)), 4)}", delta_color="inverse")
                c3.metric("Macro F1-Score", f"{rec['f1_score']*100:.2f}%")
                c4.metric("Round Latency", f"{rec['round_latency_ms']:.1f} ms", delta="Stragglers Bypassed")

            # Live Chart Update
            with chart_container.container():
                df_chart = pd.DataFrame(chart_data).set_index('Round')
                ch_col1, ch_col2 = st.columns(2)
                ch_col1.line_chart(df_chart[['MEC-AI HetFL Accuracy (%)']], color='#1a73e8')
                ch_col2.line_chart(df_chart[['MEC-AI HetFL Loss']], color='#ea4335')

            time.sleep(0.12)

        status_text.success("🎉 Simulation Completed Successfully! Model achieved high consensus accuracy.")
        st.session_state['hetfl_records'] = hetfl_records
        st.session_state['all_devices'] = all_devices
        st.session_state['clusters'] = clusters
        st.session_state['test_data'] = test_data

    else:
        st.info("💡 Adjust the sidebar parameters if needed, then click **'Run Live Training'** to begin.")


# --- TAB 2: DEVICE INSPECTOR ---
with tab2:
    st.subheader("Tier 1 & Tier 2: Dynamic Node Scoring & Straggler Detection")
    st.markdown("Algorithm 2 continuously monitors the IoT devices and calculates the **Quality Score $Q(e)$** based on both communication telemetry and AI capability.")

    server, clusters, all_devices, test_data = init_environment()

    device_rows = []
    for c in clusters:
        for d in c.devices:
            q, net, exp = c.evaluate_node_quality(d)
            status = "✅ SELECTED" if (q >= quality_thresh) else "❌ REJECTED (Straggler/Constrained)"
            device_rows.append({
                "Cluster": f"{c.cluster_id} ({c.domain_name})",
                "Device ID": d.device_id,
                "Device Type": "⚠️ Straggler" if d.is_straggler else "⚡ High Speed",
                "Bandwidth (Mbps)": f"{d.network_bandwidth:.1f}",
                "Network Latency (ms)": f"{d.latency_ms:.1f}",
                "Network Score (0-1)": f"{net:.3f}",
                "AI Expertise Score (0-1)": f"{exp:.3f}",
                "Quality Score Q(e)": f"{q:.3f}",
                "Algorithm 2 Decision": status
            })

    df_devices = pd.DataFrame(device_rows)
    st.dataframe(df_devices, use_container_width=True)

    st.markdown("### Quality Score Breakdown by Cluster")
    bar_data = df_devices.groupby(["Cluster", "Device Type"])["Quality Score Q(e)"].apply(lambda x: [float(v) for v in x].pop(0) if len(x)>0 else 0).reset_index()
    st.caption("Notice how devices flagged as stragglers receive low Network Scores and are filtered below the threshold, preventing them from stalling the entire network.")


# --- TAB 3: COMPARATIVE BASELINES ---
with tab3:
    st.subheader("Benchmark Comparison: MEC-AI HetFL vs Synchronous FedAvg")
    st.markdown("""
    **The Straggler Problem (from Slide 1):**
    In standard synchronous federated learning (like vanilla FedAvg), the central server cannot aggregate until the **slowest device** finishes.
    In our prototype, **MEC-AI HetFL** uses **Algorithm 2** to filter out stragglers and perform localized intra-cluster aggregation, dramatically slashing latency while maintaining top accuracy.
    """)

    if st.button("⚖️ Run Direct Side-by-Side Comparison", type="secondary"):
        with st.spinner("Running synchronous baseline comparison on identical Non-IID data..."):
            server, clusters, all_devices, test_data = init_environment()

            # Run MEC-AI HetFL
            hetfl_accs, hetfl_lats = [], []
            for r in range(1, 11):
                rec = server.run_federated_round(r, selection_ratio=selection_ratio, quality_threshold=quality_thresh, epochs=local_epochs, lr=learning_rate)
                hetfl_accs.append(rec['accuracy'] * 100)
                hetfl_lats.append(rec['round_latency_ms'])

            # Run Synchronous FedAvg
            fedavg_hist = run_synchronous_fedavg(all_devices, test_data, rounds=10, epochs=local_epochs, lr=learning_rate)
            fedavg_accs = [h['accuracy'] * 100 for h in fedavg_hist]
            fedavg_lats = [h['round_latency_ms'] for h in fedavg_hist]

            comp_df = pd.DataFrame({
                "Round": list(range(1, 11)),
                "MEC-AI HetFL Accuracy (%)": hetfl_accs,
                "Synchronous FedAvg Accuracy (%)": fedavg_accs,
                "MEC-AI HetFL Latency (ms)": hetfl_lats,
                "Synchronous FedAvg Latency (ms)": fedavg_lats
            }).set_index("Round")

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### 🎯 Accuracy Comparison (%)")
                st.line_chart(comp_df[["MEC-AI HetFL Accuracy (%)", "Synchronous FedAvg Accuracy (%)"]])

            with col_c2:
                st.markdown("#### ⏱️ Round Latency Comparison (ms)")
                st.line_chart(comp_df[["MEC-AI HetFL Latency (ms)", "Synchronous FedAvg Latency (ms)"]])

            st.success(f"""
            **Key Takeaway Verified:**
            - **MEC-AI HetFL Latency:** ~{np.mean(hetfl_lats):.1f} ms
            - **Synchronous FedAvg Latency:** ~{np.mean(fedavg_lats):.1f} ms
            - **Latency Reduction:** ~{((np.mean(fedavg_lats) - np.mean(hetfl_lats)) / np.mean(fedavg_lats) * 100):.1f}% faster per round due to multi-edge clustering!
            """)


# --- TAB 4: PAPER TABLE 1 BENCHMARKING ---
with tab4:
    st.subheader("State-of-the-Art Evaluation (Reproducing Table 1 from Paper)")
    st.markdown("""
    The published paper (*Scientific Reports 2024, vol. 14, art. 28746*) compared **MEC-AI HetFL** against four leading edge federated learning frameworks.
    """)

    paper_table = get_paper_benchmark_table()
    st.table(pd.DataFrame(paper_table))

    st.markdown("### Visual Benchmark Highlights")
    b_col1, b_col2, b_col3 = st.columns(3)
    
    with b_col1:
        st.metric("Top Accuracy", "98.6%", "vs 95.5% - 97.8% Baselines")
    with b_col2:
        st.metric("Lowest Loss", "0.07", "Others: 0.09 - 0.15")
    with b_col3:
        st.metric("Fastest Comm Overhead", "6 ms", "Others: 8 ms - 12 ms")


st.markdown("---")
st.caption("MEC-AI HetFL Prototype · Department of Computer Science & Engineering · Birla Institute of Technology, Mesra")
