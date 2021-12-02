REGISTRY = {}

from .basic_controller import BasicMAC
from .distri_controller import DistriMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["distri_mac"] = DistriMAC