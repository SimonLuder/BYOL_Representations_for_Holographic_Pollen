import os
import re
import yaml


def load(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
    

def update(config:dict, args: dict):
    config.update(args)


def save(config, config_file):
    with open(config_file, 'w') as stream:
        try:
            yaml.safe_dump(config, stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_ckpt_config_file(ckpt_path):
    """Get config path from checkpoint folder"""
    if os.path.isfile(ckpt_path):
        ckpt_path = os.path.dirname(ckpt_path)

    config_path = None
    pattern = re.compile(r".*config.*\.ya?ml$")

    for file in os.listdir(ckpt_path):
        if pattern.match(file):
            config_path = os.path.join(ckpt_path, file)
            print(f"[INFO] Using config file: {config_path}")
            break

    if config_path is None:
        raise FileNotFoundError(
            f"No *config*.ya?ml file found in checkpoint directory: {ckpt_path}"
        )

    return config_path


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d
