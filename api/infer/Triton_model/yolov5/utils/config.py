import yaml


def load_config():
    with open("../config/config.yaml", "r", encoding='utf-8') as f:
        dicts = yaml.safe_load(f)
    return dicts


