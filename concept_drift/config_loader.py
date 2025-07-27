import os, yaml

def load_config(path=None):
    path = path or os.getenv("DRIFT_CONFIG_PATH", "config/drift_monitor_config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
