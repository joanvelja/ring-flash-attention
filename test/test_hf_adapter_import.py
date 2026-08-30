import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def test_hf_adapter_imports_without_flash_attn_extension():
    package_dir = Path(__file__).parents[1] / "ring_flash_attn"

    ring_package = ModuleType("ring_flash_attn")
    ring_package.__path__ = [str(package_dir)]
    adapters_package = ModuleType("ring_flash_attn.adapters")
    adapters_package.__path__ = [str(package_dir / "adapters")]
    llama_module = ModuleType("ring_flash_attn.llama3_flash_attn_varlen")
    llama_module.llama3_flash_attn_varlen_func = None
    llama_module.llama3_flash_attn_prepare_cu_seqlens = None

    module_name = "ring_flash_attn.adapters.hf_adapter"
    module_path = package_dir / "adapters" / "hf_adapter.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    modules = {
        "ring_flash_attn": ring_package,
        "ring_flash_attn.adapters": adapters_package,
        "ring_flash_attn.llama3_flash_attn_varlen": llama_module,
        module_name: module,
    }
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
