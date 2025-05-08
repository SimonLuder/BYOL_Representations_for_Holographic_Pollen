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
