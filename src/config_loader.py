from yaml import load, SafeLoader

def config_loader(config_location: str):
    config_data = load(config_location, Loader=SafeLoader)    

    return config_data

