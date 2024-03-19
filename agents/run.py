
from dotenv import load_dotenv
from .utils import load_agent_run_module, load_dataset_and_test_spec

agent_to_test = "seven"

load_dotenv()

# Get the dataset and the test specification for the first dataset item
dataset, test_spec = load_dataset_and_test_spec()

# Use the function
agent_run_module = load_agent_run_module(agent_to_test)

# Now you can call a function from the loaded module, passing the test data
agent_run_module.run(dataset, test_spec)
