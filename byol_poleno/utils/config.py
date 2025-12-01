import os
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
        
    for file in os.listdir(ckpt_path):
        if file.endswith("config.yaml"):
            config_path = os.path.join(ckpt_path, file)
            print(f"[INFO] Using config file: {config_path}")
            break
        else:
            print(f"[WARNING] No config file found in {ckpt_path}.")
    return config_path