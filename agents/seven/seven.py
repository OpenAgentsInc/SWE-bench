import os
from git import Repo, GitCommandError
from colorama import Fore, Style, init
from harness_devin.types import SwebenchInstance

class Seven:
    def __init__(self, dataset: SwebenchInstance):
        self.instance_id = dataset["instance_id"]
        self.repo = dataset["repo"]
        self.base_commit = dataset["base_commit"]
        self.token = os.getenv('GITHUB_TOKEN')
        self.repo_url = f"https://github.com/{self.repo}.git"
        self.local_repo_path = self.clone_or_checkout_repo(self.repo_url, self.instance_id, self.base_commit)
        print(Fore.GREEN + f"Repository is available at {self.local_repo_path}")

    def clone_or_checkout_repo(self, repo_url, instance_id, base_commit):
        # Define the local path to clone the repository to, now using instance_id
        local_path = os.path.join(os.path.dirname(__file__), 'workspace', instance_id)

        if not os.path.isdir(os.path.join(local_path, '.git')):
            # Ensure the target directory exists
            os.makedirs(local_path, exist_ok=True)

            # Clone the repository if it does not exist
            print(Fore.BLUE + "Cloning repository...")
            repo = Repo.clone_from(repo_url, local_path, env={'GIT_TERMINAL_PROMPT': '0'})
            print(Fore.GREEN + "Repository cloned successfully.")
        else:
            print(Fore.YELLOW + "Repository already exists locally. Skipping clone.")
            repo = Repo(local_path)

        # Checkout the specified base commit
        try:
            repo.git.checkout(base_commit)
            print(Fore.GREEN + f"Checked out to commit {base_commit}.")
        except GitCommandError as e:
            print(Fore.RED + f"Failed to checkout commit {base_commit}: {e}")

        return local_path
