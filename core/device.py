"""
MEC-AI HetFL: Constrained IoT Device (Tier 1)
Simulates heterogeneous IoT edge devices with local private data, compute capacities,
and varying network latency (including straggler dynamics).
"""

import numpy as np
import time

class ConstrainedIoTDevice:
    """
    Represents an IoT device in Tier 1.
    Attributes:
        device_id (str): Unique identifier e.g., 'Device-1-3'
        cluster_id (str): Cluster membership (e.g., 'Cluster-SmartCity')
        X_train, y_train: Private local dataset D_e
        network_bandwidth (Mbps): 1.0 to 50.0 Mbps
        latency_ms (float): Base latency (e.g. 10ms for fast, 48ms+ for stragglers)
        is_straggler (bool): Whether this device simulates lagging compute or network jitter
        compute_power (float): MIPS / relative compute factor (0.2 to 1.5)
    """
    def __init__(self, device_id, cluster_id, X_train, y_train, is_straggler=False, seed=None):
        self.device_id = device_id
        self.cluster_id = cluster_id
        self.X_train = X_train
        self.y_train = y_train
        self.is_straggler = is_straggler
        
        rng = np.random.RandomState(seed if seed is not None else hash(device_id) % (2**31 - 1))
        
        # Heterogeneous device characteristics
        if is_straggler:
            self.network_bandwidth = float(rng.uniform(0.5, 2.5))     # Low bandwidth
            self.latency_ms = float(rng.uniform(40.0, 75.0))          # High latency (Straggler: 48ms+)
            self.compute_power = float(rng.uniform(0.2, 0.45))        # Constrained CPU
            self.packet_loss_rate = float(rng.uniform(0.08, 0.22))
        else:
            self.network_bandwidth = float(rng.uniform(15.0, 45.0))   # High bandwidth
            self.latency_ms = float(rng.uniform(8.0, 16.0))           # Fast: 10-12ms
            self.compute_power = float(rng.uniform(0.8, 1.4))         # Capable edge device
            self.packet_loss_rate = float(rng.uniform(0.005, 0.03))

        self.last_training_loss = 0.0
        self.last_accuracy = 0.0

    def get_network_score(self):
        """
        Calculates NetworkScore(e) normalized between 0.0 and 1.0.
        Factors: High bandwidth, low latency, low packet loss.
        """
        # Bandwidth component: min(bw / 50.0, 1.0)
        bw_score = min(self.network_bandwidth / 50.0, 1.0)
        # Latency component: 1.0 - (latency / 80.0)
        lat_score = max(0.0, 1.0 - (self.latency_ms / 80.0))
        # Reliability
        rel_score = 1.0 - self.packet_loss_rate
        return float(0.4 * bw_score + 0.4 * lat_score + 0.2 * rel_score)

    def get_expertise_score(self):
        """
        Calculates ExpertiseScore(e) normalized between 0.0 and 1.0.
        Factors: Local sample size, compute capacity, class diversity.
        """
        sample_score = min(len(self.y_train) / 100.0, 1.0)
        compute_score = min(self.compute_power / 1.5, 1.0)
        class_diversity = len(np.unique(self.y_train)) / 4.0
        return float(0.4 * sample_score + 0.35 * compute_score + 0.25 * class_diversity)

    def local_train(self, model, epochs=3, lr=0.02, simulate_delay=False):
        """
        Executes local model training on private data D_e
        According to Eq. (1):
        theta_c(t) = theta_c(t-1) - eta * nabla L(theta_c(t-1), D_e)
        Returns:
            updated_weights: dict
            duration_ms: simulated elapsed time in ms
            metrics: {'loss': loss, 'acc': acc}
        """
        # Clone model weights into local instance
        local_weights = model.get_weights()
        model.set_weights(local_weights)

        # Training loop
        losses, accs = [], []
        for _ in range(epochs):
            loss, acc = model.train_on_batch(self.X_train, self.y_train, lr=lr)
            losses.append(loss)
            accs.append(acc)

        self.last_training_loss = float(np.mean(losses))
        self.last_accuracy = float(np.mean(accs))

        # Simulated physical round duration (ms)
        compute_time = (len(self.X_train) * epochs * 0.06) / self.compute_power
        transmission_time = (0.5 / max(self.network_bandwidth, 0.1)) * 1000.0
        total_time_ms = self.latency_ms + compute_time + transmission_time

        if simulate_delay:
            time.sleep(min(total_time_ms / 1000.0, 0.05))

        return model.get_weights(), total_time_ms, {
            'loss': self.last_training_loss,
            'accuracy': self.last_accuracy,
            'samples': len(self.y_train)
        }
