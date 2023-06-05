import logging
from importlib import import_module
from types import ModuleType

# set this module-level variable before calling `get_config`
config = None


def get_config() -> ModuleType:
    if config is None:
        logging.error(f"'config' is not initialised")
    name = f"pipelines.configs.{config}"
    logging.debug(f"import config '{name}'")
    return import_module(name)


__all__ = ["config", "get_config"]
