import json
import numpy as np
import os
from git import Repo, GitCommandError
from colorama import Fore, Style, init
from harness_devin.types import SwebenchInstance
from json.decoder import JSONDecodeError
from .openai_helpers.helpers import compare_embeddings, compare_text, embed, complete, complete_code, EMBED_DIMS
from pathlib import Path

class Seven:
    def __init__(self, dataset: SwebenchInstance):
        self.instance_id = dataset["instance_id"]
        self.repo = dataset["repo"]
        self.base_commit = dataset["base_commit"]
        self.token = os.getenv('GITHUB_TOKEN')
        self.repo_url = f"https://github.com/{self.repo}.git"
        self.local_repo_path = self.clone_or_checkout_repo(self.repo_url, self.instance_id, self.base_commit)
        self.descriptions_path = Path(self.local_repo_path).parent / 'descriptions'
        os.makedirs(self.descriptions_path, exist_ok=True)  # Ensure the directory for descriptions exists
        # File extensions to include
        self.extensions = ('.md', '.js', '.jsx', '.py', '.json', '.html', '.css', '.scss', '.yml', '.yaml', '.ts', '.tsx', '.ipynb', '.c', '.cc', '.cpp', '.go', '.h', '.hpp', '.java', '.sol', '.sh', '.txt')
        # Directories to exclude from processing
        self.directory_blacklist = ('build', 'dist', '.github', 'site', 'tests')
        print(Fore.GREEN + f"Repository initialized")
        self.process_repository()
        self.process_embeddings()

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

    def load_descriptions(self):
        descriptions_file = self.descriptions_path / 'descriptions.json'
        if descriptions_file.exists():
            try:
                with open(descriptions_file, 'r') as f:
                    return json.load(f)
            except JSONDecodeError:
                print(Fore.RED + "Invalid or empty JSON. Returning an empty dictionary.")
                return {}
        return {}

    def save_descriptions(self, descriptions):
        descriptions_file = self.descriptions_path / 'descriptions.json'
        with open(descriptions_file, 'w') as f:
            json.dump(descriptions, f, indent=2)

    def generate_description(self, file_path, code):
        description_prompt = 'A short summary in plain English of the above code is:'
        extension = file_path.suffix[1:]  # Remove the dot from the extension
        # Adjust the prompt to include only the relevant part of the path
        relative_path_str = str(file_path.relative_to(self.local_repo_path.parent))
        prompt = f'File: {relative_path_str}\n\nCode:\n\n```{extension}\n{code}```\n\n{description_prompt}\n'

        # Generate the description using the 'complete' function
        description = complete(prompt)

        # If the 'complete' function returns a string directly, use it as the description
        # Otherwise, you may need to adjust based on how 'complete' returns data
        return description

    def process_repository(self):
        descriptions = self.load_descriptions()
        num_files = len(descriptions)
        repo_root_path = Path(self.local_repo_path)

        for root, dirs, files in os.walk(repo_root_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in self.directory_blacklist]  # Filter out blacklisted directories
            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.suffix not in self.extensions:
                    continue  # Skip files not in the whitelist

                # Calculate the relative path from the repository root
                relative_path = file_path.relative_to(repo_root_path)

                # Convert to string and ensure it starts with "/"
                relative_path_str = f"/{relative_path}"

                if relative_path_str in descriptions:
                    continue  # Skip if description already exists

                try:
                    with open(file_path, 'r') as file:
                        code = file.read()
                    description = self.generate_description(file_path, code)
                    descriptions[relative_path_str] = description  # Save using the relative path

                    num_files += 1
                    if num_files % 10 == 0:
                        self.save_descriptions(descriptions)
                        print(f'Saved descriptions for {num_files} files.')
                except Exception as e:
                    print(Fore.RED + f"Error processing {file_path}: {e}")

        self.save_descriptions(descriptions)
        print(Fore.GREEN + f"Processed file descriptions: {num_files}.")

    def load_embeds(self):
        embeds_path = self.descriptions_path.parent / "embeddings.npy"
        if embeds_path.exists():
            return np.load(embeds_path)
        return None

    def save_embeds(self, embeds):
        embeds_path = self.descriptions_path.parent / "embeddings.npy"
        np.save(embeds_path, embeds)

    def get_embeds(self, descriptions, batch_size=50, save=True):
        embeds_path = self.descriptions_path.parent / "embeds.npy"

        # Check if embeddings already exist to avoid re-computation
        if embeds_path.exists():
            print("Loading existing embeddings.")
            return np.load(embeds_path)

        # Initialize an empty array for embeddings
        embeds = np.empty((0, EMBED_DIMS), dtype=np.float32)
        batch = []

        # Prepare batch processing for embeddings
        for description in descriptions.values():
            batch.append(description)
            if len(batch) == batch_size:
                embeds_batch = embed(batch)  # Assume embed function returns an array of embeddings
                embeds = np.append(embeds, embeds_batch, axis=0)
                batch = []  # Clear the batch

        # Process any remaining descriptions in the last batch
        if batch:
            embeds_batch = embed(batch)  # Process the final batch
            embeds = np.append(embeds, embeds_batch, axis=0)

        # Save embeddings if requested
        if save:
            print(Fore.GREEN + f"Saving embeddings to {embeds_path}.")
            np.save(embeds_path, embeds)

        return embeds

    def process_embeddings(self):
        descriptions = self.load_descriptions()
        self.embeddings = self.get_embeds(descriptions)
        print(Fore.GREEN + f"Initialized embeddings: {len(self.embeddings)}")
