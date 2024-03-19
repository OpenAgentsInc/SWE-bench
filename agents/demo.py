import os
import subprocess
from langchain_anthropic import ChatAnthropic
from harness_devin import get_dataset, make_test_spec

# Load dotenv
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Ensure the API key is set
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")

# Get the dataset and the test specification for the first dataset item
dataset = get_dataset()

# Get the first dataset
first_dataset_item = dataset[0] # repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
# test1 = make_test_spec(first_dataset_item)  # instance_id, setup_script, prompt, eval_script

# Initialize the ChatAnthropic client
chat = ChatAnthropic(anthropic_api_key=anthropic_api_key, model='claude-3-opus-20240229', temperature=0)

# Formulate the prompt
prompt = (
    f"Please help me resolve this GitHub issue for the repository {first_dataset_item['repo']}. "
    f"The issue is: {first_dataset_item['problem_statement']}\n\n"
    "Hints:\n"
    f"{first_dataset_item['hints_text']}\n\n"
    "What would be your approach to resolve this issue?"
)

# Adjust the call to invoke with a string prompt directly
response = chat.invoke(prompt)

# Print the response from the model
print("Claude's response:", response.content)
