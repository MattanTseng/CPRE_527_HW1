from yaml import safe_load, SafeLoader

def config_loader(config_location: str):
    with open(config_location, 'r') as yaml_file:
        config_data = safe_load(yaml_file) 

    return config_data

