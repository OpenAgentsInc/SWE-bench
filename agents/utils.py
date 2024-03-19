import importlib.util
from harness_devin import get_dataset, make_test_spec
from pathlib import Path

def load_agent_run_module(agent_name):
    print("Loading agent:", agent_name)
    agent_run_path = Path(__file__).parent / agent_name / 'run.py'
    spec = importlib.util.spec_from_file_location(f"{agent_name}.run", agent_run_path)
    agent_run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_run_module)

    return agent_run_module

def load_dataset_and_test_spec():
    print("Fetching dataset")
    dataset = get_dataset()
    first_dataset_item = dataset[0] # repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
    test1 = make_test_spec(first_dataset_item)  # instance_id, setup_script, prompt, eval_script
    return first_dataset_item, test1
