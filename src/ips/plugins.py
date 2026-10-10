"""Small, explicit file-based plugin discovery shared by models and preprocessing."""

import importlib
import pkgutil


def discover(package_name: str, name_attribute: str, builder_attribute: str):
    importlib.invalidate_caches()
    package = importlib.import_module(package_name)
    found = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        has_name = hasattr(module, name_attribute)
        has_builder = hasattr(module, builder_attribute)
        if not has_name and not has_builder:
            continue
        if not (has_name and has_builder):
            raise ValueError(f"Incomplete plugin {module.__name__}: requires {name_attribute} and {builder_attribute}")
        name, builder = getattr(module, name_attribute), getattr(module, builder_attribute)
        if not isinstance(name, str) or not name or not callable(builder):
            raise ValueError(f"Invalid plugin definition in {module.__name__}")
        if name in found:
            raise ValueError(f"Duplicate plugin name {name!r} in {package_name}")
        found[name] = builder
    return found
