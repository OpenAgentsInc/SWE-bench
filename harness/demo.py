import os
import subprocess
from .dataset import get_dataset
from .scripts import make_test_spec

# Get the dataset and the test specification for the first dataset item
dataset = get_dataset()
test1 = make_test_spec(dataset[0])  # instance_id, setup_script, prompt, eval_script

# Define the path for the workspace directory relative to the current working directory
workspace_dir = os.path.join(os.getcwd(), "workspace")

# Automatically create the workspace directory if it does not exist
os.makedirs(workspace_dir, exist_ok=True)

# Define the path for the setup script file within the workspace directory
setup_script_path = os.path.join(workspace_dir, 'setup_script.sh')

# Save the setup_script to a file in the workspace directory
with open(setup_script_path, 'w') as file:
    file.write(test1.setup_script)
    print(f"setup_script saved to {setup_script_path}")

# Now, let's build the Docker image
docker_build_command = f"docker build -t swebench_setup {workspace_dir}"
subprocess.run(docker_build_command, check=True, shell=True)
print("Docker image built.")

# And run the Docker container from the image
docker_run_command = "docker run --rm swebench_setup"
subprocess.run(docker_run_command, check=True, shell=True)
print("Docker container ran with setup script.")
