# MEC-AI HetFL: Adaptive Federated Learning Prototype

**Department of Computer Science & Engineering**  
**Birla Institute of Technology, Mesra · Patna Campus**  
*B.Tech. (CSE) – V Monsoon 2026 | CSE24300 – PROJECT-1*

---

## 📌 Project Overview
This repository contains a working software prototype of **MEC-AI HetFL** (*Multi-Edge Clustered and Edge AI Heterogeneous Federated Learning*), implemented directly from the base research paper:
> **F. R. Mughal, J. He, B. Das, et al.**  
> *"Adaptive federated learning for resource-constrained IoT devices through edge intelligence and multi-edge clustering"*,  
> **Scientific Reports (Nature Portfolio)**, vol. 14, art. 28746, 2024. [doi:10.1038/s41598-024-78239-z](https://doi.org/10.1038/s41598-024-78239-z).

---

## 🏗️ 3-Tier Hierarchical Architecture

The architecture maps directly to **Figure 1 and Slide 3**:

```
                     ┌─────────────────────────────────────────┐
                     │    Tier 3: Hetero Federated Server       │
                     │  - Inter-Cluster Consensus Aggregation  │
                     │  - Global Model θ(t+1) = Σ θ_c / N_bc   │
                     └────────────────────▲────────────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
         ┌───────────────────────────────┐ ┌───────────────────────────────┐
         │ Tier 2: Edge AI Node 1        │ │ Tier 2: Edge AI Node 2        │
         │ (Smart City Cluster)          │ │ (Mobile Healthcare Cluster)   │
         │ - Algorithm 2 Quality Scoring │ │ - Algorithm 2 Quality Scoring │
         │ - Intra-Cluster FedAvg θ_c(t) │ │ - Intra-Cluster FedAvg θ_c(t) │
         └───────────────▲───────────────┘ └───────────────▲───────────────┘
                         │                                 │
                 ┌───────┴───────┐                 ┌───────┴───────┐
                 ▼               ▼                 ▼               ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │ Tier 1 Device│ │ Tier 1 Device│ │ Tier 1 Device│ │ Tier 1 Device│
         │ (Fast Node)  │ │ (Straggler)  │ │ (Fast Node)  │ │ (Non-IID)    │
         └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🧮 Mathematical Equations Implemented

| Formula | Description | Code Location |
| :--- | :--- | :--- |
| **Equation (1)** | $\theta_c(t) = \theta_c(t - 1) - \eta \nabla L(\theta_c(t - 1), D_e)$ (Device Local Gradient Update) | `core/device.py` |
| **Equation (2)** | $\theta_c(t) = \theta_c(t) - \eta \nabla L(\theta_c(t), D_c)$ (Intra-Cluster Model Aggregation & Refinement) | `core/edge_ai_node.py` |
| **Equation (3)** | $\theta(t + 1) = \frac{\sum_c \theta_c(t)}{N_{bc}}$ (Global Inter-Cluster Consensus Aggregation) | `core/hetero_server.py` |
| **Equation (4)** | $Q(e) = w_1 \cdot \text{NetworkScore}(e) + w_2 \cdot \text{ExpertiseScore}(e)$ (Algorithm 2 Dual-Factor Selection) | `core/edge_ai_node.py` |
| **Equations (5–8)** | Accuracy, Recall, Precision, and F1-Score evaluation | `core/models.py` |

---

## 📁 Repository Structure

```
mec_ai_hetfl_prototype/
├── app.py                  # Streamlit Interactive Web Application & Visual Dashboard
├── simulation.py           # Standalone CLI simulation & verification benchmark script
├── run_demo.bat            # 1-Click Windows execution script
├── core/
│   ├── __init__.py         # Package exports
│   ├── models.py           # HeteroEdgeMLP neural net with forward/backward passes & metrics
│   ├── dataset.py          # Multi-cluster dataset generator (Smart City, Healthcare, Agriculture)
│   ├── device.py           # Tier 1 Constrained IoT Devices & Straggler latency simulation
│   ├── edge_ai_node.py     # Tier 2 Edge AI Nodes with Algorithm 2 Quality Scoring
│   ├── hetero_server.py    # Tier 3 Hetero Server with Algorithm 1 Global Aggregation
│   └── baselines.py        # Comparative baselines (Synchronous FedAvg, EdgeFed, Table 1 data)
└── outputs/
    └── mec_ai_hetfl_evaluation.png   # Generated convergence & latency charts
```

---

## 🚀 How to Run the Prototype

### Option 1: Interactive Streamlit Web UI (Recommended for Presentations)
In your terminal, navigate to the folder and run:
```bash
python -m streamlit run app.py
```
*Or simply double-click `run_demo.bat`.*

**Features in the UI:**
1. **Live Training Simulator:** Watch real-time accuracy and loss converge round-by-round.
2. **Device Inspector:** See live $Q(e)$ scores for every IoT device, showing how stragglers are rejected and fast nodes accepted.
3. **Side-by-Side Comparison:** Compare MEC-AI HetFL vs Synchronous FedAvg latency and see the straggler problem solved in real time.
4. **Table 1 Replication:** Visual reproduction of the paper's comparison against EdgeFed, FedSA, FedMP, and H-DDPG.

---

### Option 2: Standalone Command-Line Simulation
Run:
```bash
python simulation.py
```
This runs the full 3-Tier simulation across all 3 domains, executes the baseline comparison, prints detailed round statistics, and saves high-resolution charts to `outputs/mec_ai_hetfl_evaluation.png`.

---

## 📊 Summary of Evaluation Results

| Metric | Proposed MEC-AI HetFL | Standard Synchronous FedAvg | Paper Target (Table 1) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **99.5%+** | 97.8% | **98.6%** |
| **Cross-Entropy Loss** | **0.08** | 0.21 | **0.07** |
| **Round Latency** | **~73 ms** | **~850 ms** *(delayed by stragglers)* | **10 s vs 18 s** |
| **Straggler Handling** | **Dynamic rejection via $Q(e)$** | Halts at slowest device barrier | Sub-round async |
