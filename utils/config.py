def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
