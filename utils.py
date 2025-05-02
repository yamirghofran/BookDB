import yaml

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        config = {} # Set config to empty dict or handle error appropriately
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = {} # Set config to empty dict or handle error appropriately
        return config
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        config = {}
        return config

