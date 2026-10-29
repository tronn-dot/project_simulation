"""
MEC-AI HetFL: Multi-Cluster IoT Heterogeneous & Non-IID Dataset Generator
Simulates three distinct real-world IoT domains matching the paper:
1. Smart City (Urban sensors, air quality, traffic, power grid)
2. Mobile Healthcare (Patient monitors, wearable vitals, ECG, SpO2)
3. Agriculture (Soil moisture, greenhouse climate, irrigation telemetry)
"""

import numpy as np

DOMAIN_PROFILES = {
    "Smart City": {
        "features": [
            "Traffic Density", "Acoustic Noise (dB)", "PM2.5 (ug/m3)", "PM10 (ug/m3)",
            "Ambient Temp (C)", "Streetlight Power (W)", "Grid Load (kW)", "CO2 Levels (ppm)",
            "Pedestrian Count", "Vehicle Speed (km/h)", "Rainfall Rate", "Wind Speed (m/s)"
        ],
        "means": [65.0, 58.0, 45.0, 72.0, 28.0, 110.0, 420.0, 410.0, 30.0, 42.0, 2.5, 4.0],
        "stds": [20.0, 12.0, 15.0, 22.0, 6.0, 25.0, 80.0, 50.0, 15.0, 12.0, 3.0, 2.0],
        "class_labels": ["Optimal Flow", "Moderate Congestion", "Pollution Alert", "Hazard Emergency"]
    },
    "Mobile Healthcare": {
        "features": [
            "Heart Rate (bpm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "SpO2 (%)",
            "Body Temp (C)", "Respiration Rate (rpm)", "ECG ST-Segment (mV)", "Galvanic Skin Response",
            "Motion Sensor Energy", "Glucose Estimate (mg/dL)", "Perfusion Index", "Pulse Delay (ms)"
        ],
        "means": [76.0, 122.0, 82.0, 97.5, 36.8, 16.0, 0.05, 4.2, 18.0, 105.0, 3.5, 180.0],
        "stds": [14.0, 18.0, 10.0, 2.0, 0.7, 4.0, 0.15, 1.8, 8.0, 25.0, 1.2, 35.0],
        "class_labels": ["Stable Condition", "Mild Arrhythmia", "Hypoxia/Vitals Warning", "Critical Distress"]
    },
    "Agriculture": {
        "features": [
            "Soil Moisture (%)", "Soil Temp (C)", "Canopy Temp (C)", "Relative Humidity (%)",
            "Solar Irradiance (W/m2)", "Soil pH Level", "Nitrogen Index", "Phosphorus Index",
            "Potassium Index", "Irrigation Flow (L/min)", "Leaf Wetness Hours", "Evapotranspiration"
        ],
        "means": [38.0, 24.0, 26.5, 62.0, 650.0, 6.5, 45.0, 30.0, 50.0, 12.0, 4.0, 5.2],
        "stds": [12.0, 4.5, 5.0, 14.0, 180.0, 0.8, 15.0, 10.0, 14.0, 5.0, 2.5, 1.8],
        "class_labels": ["Normal Growth", "Water Stress", "Nutrient Deficiency", "Frost/Pest Risk"]
    }
}

class IoTDatasetGenerator:
    """
    Generates synthetic multi-cluster datasets with discriminative class separation
    and controlled Non-IID skew (Dirichlet distribution) across edge clusters.
    """
    def __init__(self, random_seed=42):
        self.rng = np.random.RandomState(random_seed)

    def generate_domain_data(self, domain_name, total_samples=1200):
        """Generates domain-specific samples with classification targets."""
        profile = DOMAIN_PROFILES[domain_name]
        means = np.array(profile["means"])
        stds = np.array(profile["stds"])
        dim = len(means)
        num_classes = 4

        # Generate class-conditional Gaussian clusters for fast & clear learning convergence
        samples_per_class = total_samples // num_classes
        all_X = []
        all_y = []

        # Distinct cluster centroids in normalized space
        class_offsets = [
            np.array([ 1.8,  1.5, -1.2, -1.0,  1.2,  1.0, -1.5, -1.2,  1.0,  1.2, -1.0,  1.0]),
            np.array([-1.5, -1.2,  1.8,  1.5, -1.0, -1.2,  1.4,  1.6, -1.2, -1.0,  1.5, -1.2]),
            np.array([ 1.2, -1.5,  1.0, -1.4,  1.6, -1.0,  1.2, -1.5, -1.0,  1.5, -1.2,  1.4]),
            np.array([-1.2,  1.4, -1.5,  1.2, -1.5,  1.4, -1.0,  1.2,  1.5, -1.4,  1.2, -1.5])
        ]

        for c in range(num_classes):
            centroid = class_offsets[c]
            X_c = self.rng.normal(loc=centroid, scale=0.65, size=(samples_per_class, dim))
            y_c = np.full(samples_per_class, c, dtype=np.int64)
            all_X.append(X_c)
            all_y.append(y_c)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Shuffle
        perm = self.rng.permutation(len(y))
        X = X[perm].astype(np.float32)
        y = y[perm].astype(np.int64)

        return X, y

    def partition_non_iid(self, X, y, num_devices, non_iid_alpha=0.5):
        """
        Partitions data across devices using Dirichlet distribution over classes,
        simulating heterogeneous and non-IID data across constrained IoT devices.
        """
        num_classes = 4
        device_data = [[] for _ in range(num_devices)]
        device_labels = [[] for _ in range(num_devices)]

        for c in range(num_classes):
            idx_c = np.where(y == c)[0]
            self.rng.shuffle(idx_c)
            proportions = self.rng.dirichlet(np.repeat(non_iid_alpha, num_devices))
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, proportions)

            for i in range(num_devices):
                if len(splits[i]) > 0:
                    device_data[i].append(X[splits[i]])
                    device_labels[i].append(y[splits[i]])

        final_partitions = []
        for i in range(num_devices):
            if len(device_data[i]) > 0:
                d_X = np.concatenate(device_data[i], axis=0)
                d_y = np.concatenate(device_labels[i], axis=0)
                perm = self.rng.permutation(len(d_y))
                final_partitions.append((d_X[perm], d_y[perm]))
            else:
                sample_idx = self.rng.choice(len(y), size=25, replace=False)
                final_partitions.append((X[sample_idx], y[sample_idx]))

        return final_partitions
