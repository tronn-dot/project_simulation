"""
MEC-AI HetFL: Standalone Simulation & Verification Runner
Demonstrates the complete 3-Tier Multi-Edge Clustered Federated Learning Pipeline
Paper: Mughal et al., Scientific Reports (Nature Portfolio), 2024

Executes:
1. Synthetic Non-IID Multi-Domain Dataset Generation (Smart City, Healthcare, Agriculture)
2. Constrained IoT Edge Device Initialization with Straggler Injection
3. Tier 2 Edge AI Node Setup with Dual-Factor Quality Scoring (Algorithm 2)
4. Tier 3 Hetero Federated Server Inter-Cluster Aggregation (Algorithm 1)
5. Comparative Benchmark (MEC-AI HetFL vs Synchronous FedAvg)
6. Performance Chart Exporting (PNG)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.dataset import IoTDatasetGenerator, DOMAIN_PROFILES
from core.device import ConstrainedIoTDevice
from core.edge_ai_node import EdgeAINode
from core.hetero_server import HeteroFederatedServer
from core.baselines import run_synchronous_fedavg, get_paper_benchmark_table

def build_simulation_environment(devices_per_cluster=5, straggler_prob=0.3, w1=0.5, w2=0.5, seed=42):
    """Sets up the 3-Tier MEC-AI HetFL architecture."""
    gen = IoTDatasetGenerator(random_seed=seed)
    domains = ["Smart City", "Mobile Healthcare", "Agriculture"]
    
    clusters = []
    all_devices = []
    all_test_X = []
    all_test_y = []

    print("=" * 75)
    print(" [1/4] INITIALIZING 3-TIER HIERARCHICAL MEC-AI TOPOLOGY")
    print("=" * 75)

    for i, domain in enumerate(domains):
        # 1. Generate domain dataset
        X, y = gen.generate_domain_data(domain, total_samples=1000)
        
        # Split train (80%) and test (20%)
        split_idx = int(0.8 * len(y))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        all_test_X.append(X_test)
        all_test_y.append(y_test)

        # 2. Partition Non-IID data across devices
        partitions = gen.partition_non_iid(X_train, y_train, devices_per_cluster, non_iid_alpha=0.6)

        # 3. Initialize Tier 2 Edge AI Node
        cluster_node = EdgeAINode(
            cluster_id=f"Cluster-{i+1}",
            domain_name=domain,
            w1=w1,
            w2=w2,
            seed=seed + i
        )
        # Cluster validation subset
        cluster_node.set_cluster_data(X_test[:50], y_test[:50])

        # 4. Initialize Tier 1 Constrained IoT Devices
        rng = np.random.RandomState(seed + i * 10)
        for d_idx, (d_X, d_y) in enumerate(partitions):
            # Inject straggler with probability
            is_straggler = (rng.rand() < straggler_prob)
            dev_id = f"Node-{i+1}-{d_idx+1}"
            dev = ConstrainedIoTDevice(
                device_id=dev_id,
                cluster_id=cluster_node.cluster_id,
                X_train=d_X,
                y_train=d_y,
                is_straggler=is_straggler,
                seed=seed + d_idx
            )
            cluster_node.register_device(dev)
            all_devices.append(dev)

        clusters.append(cluster_node)
        print(f" -> Configured {cluster_node.cluster_id} ({domain}): {len(cluster_node.devices)} IoT devices")

    # Global held-out test data
    global_X_test = np.concatenate(all_test_X, axis=0)
    global_y_test = np.concatenate(all_test_y, axis=0)

    # 5. Initialize Tier 3 Hetero Federated Server
    server = HeteroFederatedServer(clusters=clusters, seed=seed)
    server.set_global_test_data(global_X_test, global_y_test)
    server.broadcast_global_model()

    return server, clusters, all_devices, (global_X_test, global_y_test)


def run_full_simulation(num_rounds=12):
    """Executes the simulation and compares against baseline."""
    server, clusters, all_devices, test_data = build_simulation_environment(
        devices_per_cluster=6,
        straggler_prob=0.35,
        w1=0.5,
        w2=0.5,
        seed=100
    )

    print("\n" + "=" * 75)
    print(" [2/4] EXECUTING MEC-AI HetFL FEDERATED TRAINING ROUNDS")
    print("       (Dynamic Dual-Factor Selection + Intra/Inter-Cluster Aggregation)")
    print("=" * 75)
    print(f"{'Round':<6} | {'Accuracy':<10} | {'Loss':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<10} | {'Latency (ms)':<12}")
    print("-" * 75)

    hetfl_history = []
    for r in range(1, num_rounds + 1):
        record = server.run_federated_round(
            round_idx=r,
            selection_ratio=0.70,
            quality_threshold=0.45,
            epochs=3,
            lr=0.03
        )
        hetfl_history.append(record)
        print(f"{record['round']:<6} | {record['accuracy']*100:>8.2f}% | {record['loss']:>8.4f} | "
              f"{record['precision']*100:>8.2f}% | {record['recall']*100:>6.2f}% | "
              f"{record['f1_score']*100:>8.2f}% | {record['round_latency_ms']:>10.2f} ms")

    print("\n" + "=" * 75)
    print(" [3/4] RUNNING COMPARATIVE BASELINE: STANDARD SYNCHRONOUS FedAvg")
    print("       (Suffering from Straggler Barrier Delay as in Slide 1)")
    print("=" * 75)
    
    fedavg_history = run_synchronous_fedavg(
        devices=all_devices,
        global_test_data=test_data,
        rounds=num_rounds,
        epochs=3,
        lr=0.03,
        seed=100
    )

    for r_data in fedavg_history[-5:]:
        print(f"Round {r_data['round']:>2} | Acc: {r_data['accuracy']*100:.2f}% | "
              f"Loss: {r_data['loss']:.4f} | Latency: {r_data['round_latency_ms']:.2f} ms (Straggler delayed!)")

    print("\n" + "=" * 75)
    print(" [4/4] GENERATING AND SAVING VERIFICATION PLOTS")
    print("=" * 75)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    plot_results(hetfl_history, fedavg_history, clusters, output_dir)
    print(f" -> Plots saved successfully to: {output_dir}")

    # Print Table 1 comparison
    print("\n" + "=" * 75)
    print(" PAPER TABLE 1 REFERENCE COMPARISON")
    print("=" * 75)
    table_data = get_paper_benchmark_table()
    for row in table_data:
        print(f"{row['Method']:<22} | Acc: {row['Accuracy (100th)']} | Loss: {row['Loss (100th)']} | "
              f"Time: {row['Training Time']} | Comm: {row['Comm Overhead']}")

    return hetfl_history, fedavg_history


def plot_results(hetfl_hist, fedavg_hist, clusters, output_dir):
    """Generates comparison charts matching paper Figures 7, 8, 11."""
    rounds = [h['round'] for h in hetfl_hist]
    hetfl_acc = [h['accuracy'] * 100 for h in hetfl_hist]
    fedavg_acc = [h['accuracy'] * 100 for h in fedavg_hist]

    hetfl_lat = [h['round_latency_ms'] for h in hetfl_hist]
    fedavg_lat = [h['round_latency_ms'] for h in fedavg_hist]

    hetfl_loss = [h['loss'] for h in hetfl_hist]
    fedavg_loss = [h['loss'] for h in fedavg_hist]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy Convergence
    axs[0].plot(rounds, hetfl_acc, 'o-', color='#1a73e8', linewidth=2.5, label='MEC-AI HetFL (Proposed)')
    axs[0].plot(rounds, fedavg_acc, 's--', color='#ea4335', linewidth=2, label='Standard FedAvg (Sync)')
    axs[0].set_title('Global Test Accuracy Convergence (%)', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Federated Round')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_ylim([40, 102])
    axs[0].legend(loc='lower right')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Round Latency (Mitigating Straggler Effect)
    axs[1].plot(rounds, hetfl_lat, 'o-', color='#34a853', linewidth=2.5, label='MEC-AI HetFL (Filtered)')
    axs[1].plot(rounds, fedavg_lat, 'x--', color='#ea4335', linewidth=2, label='Standard FedAvg (Straggler Barrier)')
    axs[1].set_title('Round Completion Latency (ms)', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Federated Round')
    axs[1].set_ylabel('Latency (ms)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Quality Score Q(e) Distribution
    cluster_names = []
    avg_quality_scores = []
    for c in clusters:
        cluster_names.append(c.domain_name)
        scores = [c.evaluate_node_quality(d)[0] * 100 for d in c.devices]
        avg_quality_scores.append(np.mean(scores))

    axs[2].bar(cluster_names, avg_quality_scores, color=['#4285f4', '#fbbc05', '#34a853'], width=0.55)
    axs[2].set_title('Avg Node Quality Score Q(e) by Cluster (%)', fontsize=12, fontweight='bold')
    axs[2].set_ylabel('Quality Score (%)')
    axs[2].set_ylim([0, 100])
    for i, v in enumerate(avg_quality_scores):
        axs[2].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    axs[2].grid(True, linestyle='--', alpha=0.6, axis='y')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "mec_ai_hetfl_evaluation.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f" -> Comprehensive chart exported to: {chart_path}")

if __name__ == "__main__":
    run_full_simulation(num_rounds=10)
