"""
MEC-AI HetFL: MEC-in-Edge AI Layer (Tier 2)
Implements Edge AI cluster nodes, intra-cluster model aggregation,
and Algorithm 2 (Edge Node Selection Algorithm with dual-factor Quality Score Q).
"""

import numpy as np
from core.models import HeteroEdgeMLP

class EdgeAINode:
    """
    Tier 2 Edge AI Node managing a localized cluster of IoT devices
    (e.g., Smart City Cluster, Mobile Healthcare Cluster, Agriculture Cluster).
    
    Responsibilities:
    1. Score each child IoT device using Algorithm 2:
       Q(e) = w1 * NetworkScore(e) + w2 * ExpertiseScore(e)
    2. Dynamically select top-performing nodes to eliminate stragglers.
    3. Perform Intra-cluster Federated Aggregation (Eq. 1 & 2).
    4. Maintain local model individuality theta_c and perform asynchronous retraining.
    """
    def __init__(self, cluster_id, domain_name, w1=0.5, w2=0.5, seed=42):
        self.cluster_id = cluster_id
        self.domain_name = domain_name
        self.w1 = w1  # Weight for Network Score
        self.w2 = w2  # Weight for AI Expertise Score
        self.devices = []
        self.local_model = HeteroEdgeMLP(input_dim=12, hidden_dim=24, output_dim=4, seed=seed)
        self.cluster_validation_data = None
        self.history = []

    def set_weights(self, weights):
        """Update cluster model weights (e.g., from global server broadcast)."""
        self.local_model.set_weights(weights)

    def get_weights(self):
        """Retrieve current cluster model weights theta_c."""
        return self.local_model.get_weights()

    def register_device(self, device):
        """Registers a Tier 1 IoT device to this edge cluster."""
        self.devices.append(device)

    def set_cluster_data(self, X_val, y_val):
        """Stores cluster-level reference data D_c for intra-cluster refinement."""
        self.cluster_validation_data = (X_val, y_val)

    def evaluate_node_quality(self, device):
        """
        Algorithm 2:
        Q(e) = w1 * NetworkScore(e) + w2 * ExpertiseScore(e)  [Eq. 4]
        Returns:
            q_score (float): Quality score between 0.0 and 1.0
            net_score (float): Network condition score
            exp_score (float): AI expertise score
        """
        net_score = device.get_network_score()
        exp_score = device.get_expertise_score()
        q_score = (self.w1 * net_score) + (self.w2 * exp_score)
        return float(q_score), float(net_score), float(exp_score)

    def select_active_nodes(self, selection_ratio=0.75, quality_threshold=0.45):
        """
        Algorithm 2: Edge Node Selection Process
        Scores candidates, filters out sub-threshold nodes (stragglers),
        and returns sorted list of selected devices.
        """
        candidates = []
        for dev in self.devices:
            q, net, exp = self.evaluate_node_quality(dev)
            candidates.append({
                'device': dev,
                'q_score': q,
                'net_score': net,
                'exp_score': exp,
                'is_straggler': dev.is_straggler
            })

        # Sort descending by quality score Q
        candidates.sort(key=lambda x: x['q_score'], reverse=True)

        # Filter nodes meeting quality requirements
        qualified = [c for c in candidates if c['q_score'] >= quality_threshold]
        
        # Take top selection_ratio fraction
        k = max(1, int(len(qualified) * selection_ratio))
        selected = qualified[:k]

        return selected, candidates

    def run_intra_cluster_round(self, selection_ratio=0.75, quality_threshold=0.45, epochs=3, lr=0.02):
        """
        Executes one intra-cluster federated round:
        1. Select best nodes via Algorithm 2.
        2. Parallel local training on selected devices.
        3. Intra-cluster model aggregation: theta_c(t) = Average(theta_c,e).
        4. Edge AI refinement on cluster data D_c (Eq. 2).
        """
        selected_records, all_candidates = self.select_active_nodes(
            selection_ratio=selection_ratio,
            quality_threshold=quality_threshold
        )

        if not selected_records:
            # Fallback if strict threshold filtered all
            selected_records = all_candidates[:1]

        device_weights = []
        latencies = []
        sample_weights = []

        # Local training on selected nodes
        for rec in selected_records:
            dev = rec['device']
            # Send current cluster model theta_c to device
            weights, dur_ms, meta = dev.local_train(self.local_model, epochs=epochs, lr=lr)
            device_weights.append(weights)
            latencies.append(dur_ms)
            sample_weights.append(meta['samples'])

        # Aggregate intra-cluster weights (FedAvg among chosen cluster nodes)
        total_samples = sum(sample_weights)
        new_w = {k: np.zeros_like(v) for k, v in self.local_model.get_weights().items()}

        for i, w in enumerate(device_weights):
            alpha = sample_weights[i] / total_samples
            for k in new_w.keys():
                new_w[k] += alpha * w[k]

        self.local_model.set_weights(new_w)

        # Refine on cluster node reference data D_c according to Eq. (2)
        if self.cluster_validation_data is not None:
            X_c, y_c = self.cluster_validation_data
            self.local_model.train_on_batch(X_c, y_c, lr=lr * 0.5)

        # In asynchronous edge AI, the round time is determined by the slowest of the *selected* nodes.
        # Since stragglers were filtered by Algorithm 2, round time remains low!
        effective_round_latency = max(latencies) if latencies else 0.0

        round_summary = {
            'cluster_id': self.cluster_id,
            'domain': self.domain_name,
            'selected_count': len(selected_records),
            'total_devices': len(self.devices),
            'round_latency_ms': effective_round_latency,
            'candidates': all_candidates,
            'selected_devices': [r['device'].device_id for r in selected_records]
        }
        self.history.append(round_summary)
        return round_summary
