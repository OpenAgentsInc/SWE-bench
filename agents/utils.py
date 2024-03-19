import importlib.util
import pickle
from harness_devin import get_dataset, make_test_spec
from pathlib import Path

CACHE_FILE = Path(__file__).parent / 'dataset_cache.pkl'

def load_agent_run_module(agent_name):
    print("Loading agent:", agent_name)
    agent_run_path = Path(__file__).parent / agent_name / 'run.py'
    spec = importlib.util.spec_from_file_location(f"{agent_name}.run", agent_run_path)
    agent_run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_run_module)

    return agent_run_module

def load_dataset_and_test_spec():
    # Check if the cache file exists
    if CACHE_FILE.is_file():
        print("Using cached dataset")
        # Read the cache file
        with open(CACHE_FILE, 'rb') as cache_file:
            dataset = pickle.load(cache_file)
    else:
        print("Fetching dataset")
        # Fetch the dataset and cache it
        dataset = get_dataset()
        with open(CACHE_FILE, 'wb') as cache_file:
            pickle.dump(dataset, cache_file)

    first_dataset_item = dataset[0]
    test1 = make_test_spec(first_dataset_item)
    return first_dataset_item, test1
