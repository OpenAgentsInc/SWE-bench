from harness_devin import get_dataset, make_test_spec
from dotenv import load_dotenv
from .utils import load_agent_run_module

agent_to_test = "seven"

load_dotenv()

# Get the dataset and the test specification for the first dataset item
print("Fetching dataset")
dataset = get_dataset()
first_dataset_item = dataset[0] # repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
test1 = make_test_spec(first_dataset_item)  # instance_id, setup_script, prompt, eval_script

# Use the function
agent_run_module = load_agent_run_module(agent_to_test)

# Now you can call a function from the loaded module, passing the test data
agent_run_module.run(test1)
