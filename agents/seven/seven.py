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
    extensions = ('.js', '.jsx', '.py', '.md', '.json', '.html', '.css', '.yml', '.yaml', '.ts', '.tsx', '.ipynb', '.c', '.cc', '.cpp', '.go', '.h', '.hpp', '.java', '.sol', '.sh', '.txt')
    directory_blacklist = ('build', 'dist', '.github')

    def __init__(self, dataset: SwebenchInstance):
        self.instance_id = dataset["instance_id"]
        self.repo = dataset["repo"]
        self.base_commit = dataset["base_commit"]
        self.token = os.getenv('GITHUB_TOKEN')
        self.repo_url = f"https://github.com/{self.repo}.git"

        # This path is the root for this instance, including descriptions, embeddings, and the cloned repo.
        self.instance_path = Path(__file__).parent / 'workspace' / self.instance_id
        os.makedirs(self.instance_path, exist_ok=True)  # Ensure the instance directory exists

        # The local_repo_path is now directly under instance_path. The repo's natural folder is created by git clone.
        self.local_repo_path = self.instance_path / self.repo.split('/')[-1]

        # Adjustments are not needed for descriptions and embeddings paths since they're correctly placed
        self.descriptions_path = self.instance_path / 'descriptions.json'
        self.embeddings_path = self.instance_path / "embeddings.npy"

        print(Fore.GREEN + f"Repository initialization setup at {self.instance_path.relative_to(Path(__file__).parent)}")
        self.clone_or_checkout_repo(self.repo_url, self.base_commit)
        self.process_repository()
        self.process_embeddings()

    def clone_or_checkout_repo(self, repo_url, base_commit):
        if not self.local_repo_path.joinpath('.git').exists():
            print(Fore.BLUE + "Cloning repository...")
            Repo.clone_from(repo_url, str(self.local_repo_path), env={'GIT_TERMINAL_PROMPT': '0'})
            print(Fore.GREEN + "Repository cloned successfully into " + str(self.local_repo_path.relative_to(Path(__file__).parent)))
        else:
            print(Fore.YELLOW + "Repository already exists locally at " + str(self.local_repo_path.relative_to(Path(__file__).parent)) + ". Skipping clone.")

        # Checkout the specified base commit
        repo = Repo(str(self.local_repo_path))
        try:
            repo.git.checkout(base_commit)
            print(Fore.GREEN + f"Checked out to commit {base_commit}.")
        except GitCommandError as e:
            print(Fore.RED + f"Failed to checkout commit {base_commit}: {e}")

    def load_descriptions(self):
        descriptions_file = self.descriptions_path
        if descriptions_file.exists():
            try:
                with open(descriptions_file, 'r') as f:
                    return json.load(f)
            except JSONDecodeError:
                print(Fore.RED + "Invalid or empty JSON. Returning an empty dictionary.")
                return {}
        return {}

    def save_descriptions(self, descriptions):
        with open(self.descriptions_path, 'w') as f:
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
        # Load existing descriptions
        existing_descriptions = self.load_descriptions()
        print(Fore.YELLOW + f"Starting with {len(existing_descriptions)} existing descriptions.")

        # Counter for new descriptions since the last save
        new_descriptions_since_last_save = 0

        # Iterate over each file in the repository
        total_files_processed = 0
        for root, dirs, files in os.walk(self.local_repo_path, topdown=True):
            # Filter out blacklisted directories
            dirs[:] = [d for d in dirs if d not in self.directory_blacklist]

            for file_name in files:
                file_path = Path(root) / file_name

                # Calculate the relative path from the repository root
                # Important: Ensure this matches exactly how keys are stored in descriptions.json
                relative_path_str = '/' + str(file_path.relative_to(self.local_repo_path)).replace("\\", "/")

                # Skip files not matching the extensions whitelist or already described
                if file_path.suffix not in self.extensions or relative_path_str in existing_descriptions:
                    continue

                try:
                    with open(file_path, 'r') as file:
                        code = file.read()
                    # Generate a new description
                    description = self.generate_description(file_path, code)
                    existing_descriptions[relative_path_str] = description

                    new_descriptions_since_last_save += 1
                    total_files_processed += 1

                    # Save every 10 new descriptions
                    if new_descriptions_since_last_save >= 10:
                        self.save_descriptions(existing_descriptions)
                        print(Fore.GREEN + f"Saved intermediate descriptions after processing {total_files_processed} files.")
                        new_descriptions_since_last_save = 0

                except Exception as e:
                    print(Fore.RED + f"Error processing {relative_path_str}: {e}")

        # Final save to capture any remaining new descriptions
        if new_descriptions_since_last_save > 0:
            self.save_descriptions(existing_descriptions)
            print(Fore.GREEN + f"Final save of descriptions after processing all files.")

        print(Fore.GREEN + f"Total files processed: {total_files_processed}.")
        print(Fore.GREEN + f"Total file descriptions now available: {len(existing_descriptions)}.")


    def load_embeds(self):
        if self.embeddings_path.exists():
            return np.load(self.embeddings_path)
        return None

    def save_embeds(self, embeds):
        np.save(self.embeddings_path, embeds)

    def get_embeds(self, descriptions, batch_size=50, save=True):
        # Path to save the embeddings
        embeds_path = self.embeddings_path

        # If embeddings file already exists, load and return
        if embeds_path.exists():

            # If # of embeddings of the existing file does not match the # of descriptions, re-generate
            if len(descriptions) != len(np.load(embeds_path)):
                print(Fore.YELLOW + "Number of descriptions has changed. Re-generating embeddings.")
            else:
                print("Loading existing embeddings.")
                embeddings = np.load(embeds_path)
                print(f"Loaded {len(embeddings)} embeddings.")
                return embeddings

        # Initialize an empty list to collect embeddings
        embeddings = []

        # Prepare descriptions in batches for embedding
        description_texts = list(descriptions.values())
        for i in range(0, len(description_texts), batch_size):
            batch_texts = description_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{len(description_texts)//batch_size + 1}")
            batch_embeddings = embed(batch_texts)  # Assuming this returns a list of numpy arrays
            embeddings.extend(batch_embeddings)

        # Convert list of embeddings to a single numpy array
        embeddings = np.array(embeddings)

        # Save embeddings if requested
        if save and embeddings.size > 0:
            print(f"Saving {len(embeddings)} embeddings to {embeds_path}.")
            np.save(embeds_path, embeddings)
        elif not embeddings.size > 0:
            print("No embeddings were generated.")

        return embeddings

    def process_embeddings(self):
        descriptions = self.load_descriptions()
        print(Fore.YELLOW + f"Loaded {len(descriptions)} descriptions.")
        self.embeddings = self.get_embeds(descriptions)
        print(Fore.GREEN + f"Initialized embeddings: {len(self.embeddings)}")
