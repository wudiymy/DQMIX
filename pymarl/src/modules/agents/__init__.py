REGISTRY = {}

from .rnn_agent import RNNAgent
from .distri_rnn_agent import DRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["drnn"] = DRNNAgent