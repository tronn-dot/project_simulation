"""
MEC-AI HetFL Prototype Core Package
"""
from core.models import HeteroEdgeMLP
from core.dataset import IoTDatasetGenerator, DOMAIN_PROFILES
from core.device import ConstrainedIoTDevice
from core.edge_ai_node import EdgeAINode
from core.hetero_server import HeteroFederatedServer
from core.baselines import run_synchronous_fedavg, run_edgefed_baseline, get_paper_benchmark_table

__all__ = [
    'HeteroEdgeMLP',
    'IoTDatasetGenerator',
    'DOMAIN_PROFILES',
    'ConstrainedIoTDevice',
    'EdgeAINode',
    'HeteroFederatedServer',
    'run_synchronous_fedavg',
    'run_edgefed_baseline',
    'get_paper_benchmark_table'
]
