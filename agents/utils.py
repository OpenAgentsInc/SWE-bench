import importlib.util
from pathlib import Path

def load_agent_run_module(agent_name):
    print("Loading agent:", agent_name)

    # Construct the path to the run.py file within the agent's folder
    agent_run_path = Path(__file__).parent / agent_name / 'run.py'

    # Load the module
    spec = importlib.util.spec_from_file_location(f"{agent_name}.run", agent_run_path)
    agent_run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_run_module)

    return agent_run_module
