"""
MEC-AI HetFL: Hetero Federated Server (Tier 3)
Implements Algorithm 1 (Inter-cluster global aggregation, consensus, and broadcast)
along with evaluation metrics matching Equations (5), (6), (7), and (8).
"""

import numpy as np
from core.models import HeteroEdgeMLP

class HeteroFederatedServer:
    """
    Tier 3 Central Hetero Server for MEC-AI HetFL.
    Orchestrates inter-cluster communication, global federated aggregation,
    and consensus across multi-edge clusters.
    """
    def __init__(self, clusters=None, seed=42):
        self.clusters = clusters or []
        self.global_model = HeteroEdgeMLP(input_dim=12, hidden_dim=24, output_dim=4, seed=seed)
        self.global_test_data = None
        self.round_history = []

    def register_cluster(self, cluster_node):
        """Registers a Tier 2 Edge AI Node."""
        self.clusters.append(cluster_node)

    def set_global_test_data(self, X_test, y_test):
        """Sets held-out multi-cluster test dataset for objective global validation."""
        self.global_test_data = (X_test, y_test)

    def broadcast_global_model(self):
        """Broadcasts current global model theta(t) to all Tier 2 Edge AI nodes."""
        global_w = self.global_model.get_weights()
        for cluster in self.clusters:
            cluster.set_weights(global_w)

    def aggregate_inter_cluster(self):
        """
        Inter-cluster Model Aggregation:
        theta(t+1) = (sum_c theta_c(t)) / N_bc  [Eq. 3]
        Averages model weights across N_bc active edge clusters.
        """
        N_bc = len(self.clusters)
        if N_bc == 0:
            return

        # Initialize accumulator with zeros
        aggregated_weights = {k: np.zeros_like(v) for k, v in self.global_model.get_weights().items()}

        for cluster in self.clusters:
            cw = cluster.get_weights()
            for k in aggregated_weights.keys():
                aggregated_weights[k] += cw[k]

        # Divide by N_bc (arithmetic consensus mean)
        for k in aggregated_weights.keys():
            aggregated_weights[k] /= float(N_bc)

        self.global_model.set_weights(aggregated_weights)
        return aggregated_weights

    def run_federated_round(self, round_idx, selection_ratio=0.75, quality_threshold=0.45, epochs=3, lr=0.02):
        """
        Executes a complete MEC-AI HetFL training round:
        1. Edge AI nodes run intra-cluster rounds (Algorithm 2 node selection & training).
        2. Tier 3 server aggregates edge cluster weights (Algorithm 1, Eq. 3).
        3. Global server broadcasts updated weights back to all clusters.
        4. Performance metrics are recorded.
        """
        cluster_summaries = []
        cluster_latencies = []

        # 1. Intra-cluster training & aggregation across all edge clusters
        for cluster in self.clusters:
            summary = cluster.run_intra_cluster_round(
                selection_ratio=selection_ratio,
                quality_threshold=quality_threshold,
                epochs=epochs,
                lr=lr
            )
            cluster_summaries.append(summary)
            cluster_latencies.append(summary['round_latency_ms'])

        # 2. Inter-cluster global aggregation
        self.aggregate_inter_cluster()

        # 3. Broadcast updated global model back to edge clusters
        self.broadcast_global_model()

        # 4. Global evaluation against test data
        eval_metrics = {}
        if self.global_test_data is not None:
            X_test, y_test = self.global_test_data
            eval_metrics = self.global_model.evaluate(X_test, y_test)

        # Total round time is maximum cluster latency + network inter-cluster transfer (~6ms as in Table 1)
        inter_cluster_comm_ms = 6.0  # matches Table 1 empirical overhead
        total_round_latency = max(cluster_latencies) + inter_cluster_comm_ms

        round_record = {
            'round': round_idx,
            'accuracy': eval_metrics.get('accuracy', 0.0),
            'loss': eval_metrics.get('loss', 0.0),
            'precision': eval_metrics.get('precision', 0.0),
            'recall': eval_metrics.get('recall', 0.0),
            'f1_score': eval_metrics.get('f1_score', 0.0),
            'round_latency_ms': total_round_latency,
            'cluster_summaries': cluster_summaries
        }
        self.round_history.append(round_record)
        return round_record
