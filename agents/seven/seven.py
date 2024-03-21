import datetime
import json
import numpy as np
import openai
import os
import re
from langchain_anthropic import ChatAnthropic
from git import Repo, GitCommandError
from colorama import Fore, Style, init
from harness_devin.types import SwebenchInstance
from json.decoder import JSONDecodeError
from .openai_helpers.helpers import compare_embeddings, compare_text, embed, complete, complete_code, EMBED_DIMS
from pathlib import Path

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Seven:
    extensions = ('.js', '.jsx', '.py', '.md', '.json', '.html', '.css', '.yml', '.yaml', '.ts', '.tsx', '.ipynb', '.c', '.cc', '.cpp', '.go', '.h', '.hpp', '.java', '.sol', '.sh', '.txt')
    directory_blacklist = ('build', 'dist', '.github')

    def __init__(self, dataset: SwebenchInstance):
        self.dataset = dataset
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

        print(Fore.BLUE + f"Repo initialized at {self.instance_path.relative_to(Path(__file__).parent)}")
        self.clone_or_checkout_repo(self.repo_url, self.base_commit)
        self.process_repository()
        self.process_embeddings()

    def clone_or_checkout_repo(self, repo_url, base_commit):
        if not self.local_repo_path.joinpath('.git').exists():
            print(Fore.BLUE + "Cloning repository...")
            Repo.clone_from(repo_url, str(self.local_repo_path), env={'GIT_TERMINAL_PROMPT': '0'})
            print(Fore.GREEN + "Repo cloned successfully into " + str(self.local_repo_path.relative_to(Path(__file__).parent)))
        else:
            print(Fore.BLUE + "Using existing repo at " + str(self.local_repo_path.relative_to(Path(__file__).parent)))

        # Checkout the specified base commit
        repo = Repo(str(self.local_repo_path))
        try:
            repo.git.checkout(base_commit)
            print(Fore.BLUE + f"Checked out to commit {base_commit}.")
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
        print(Fore.BLUE + f"Starting with {len(existing_descriptions)} existing descriptions.")

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

        print(Fore.BLUE + f"Total files processed: {total_files_processed}.")
        print(Fore.BLUE + f"Total file descriptions now available: {len(existing_descriptions)}.")


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
                print(Fore.BLUE + "Loading existing embeddings.")
                embeddings = np.load(embeds_path)
                print(Fore.BLUE + f"Loaded {len(embeddings)} embeddings.")
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
        print(Fore.BLUE + f"Loaded {len(descriptions)} descriptions.")
        self.embeddings = self.get_embeds(descriptions)

    def edit_file(self, file_path, problem_statement):
        # Read the original content of the file
        with open(self.local_repo_path / file_path.strip("/"), 'r') as file:
            file_content = file.read()

        # Construct a prompt to identify the code block that needs changes
        prompt_for_identifying_change = (
            f"Given the problem statement:\n\n{problem_statement}\n\n"
            f"And the content of the file {file_path}:\n\n```python\n{file_content}\n```\n\n"
            "Identify which code block needs to be changed (mark it up with \"Before:\") "
            "and suggest the change (mark it up with \"After:\"). "
            "Make your change match the coding style of the original file."
        )
        print(f"Prompt for identifying change in {file_path}:")
        print(prompt_for_identifying_change)

        # Use the custom 'complete' function to send the prompt and get the response
        change_suggestion = complete(prompt_for_identifying_change)

        if "Before:" not in change_suggestion or "After:" not in change_suggestion:
            print("Warning: incorrect output format or no changes identified.")
            return

        # Attempt a more granular approach to matching and replacing code
        try:
            # Extract the 'Before' and 'After' blocks and remove Markdown code block syntax
            before_and_after = change_suggestion.split("Before:", 1)[1]
            before, after = before_and_after.split("After:", 1)
            before = before.strip().replace("```python", "").replace("```", "").strip()
            after = after.strip().replace("```python", "").replace("```", "").strip()

            # Attempt to replace the 'Before' block with the 'After' block in the file content
            if before in file_content:
                new_file_content = file_content.replace(before, after)
                with open(self.local_repo_path / file_path.strip("/"), 'w') as file:
                    file.write(new_file_content)
                print(f"Changes applied successfully to {file_path}.")
            else:
                # If direct match fails, attempt normalization
                normalized_before = ' '.join(before.split())
                normalized_file_content = ' '.join(file_content.split())
                if normalized_before in normalized_file_content:
                    # This part is tricky; normalization can mess up the exact replacement needed
                    # A robust solution would require parsing or more sophisticated string manipulation
                    print("Found match after normalization, but exact replacement might be tricky.")
                else:
                    raise ValueError("Normalized content also does not match.")
        except ValueError as e:
            # Debugging information before throwing an exception
            print("Debugging Information:")
            print(f"File Path: {file_path}")
            print("Expected 'Before' Block:")
            print(before)
            print("Change Suggestion Received:")
            print(change_suggestion)
            print(e)
            raise Exception("Cannot locate `Before` block in the file content.")





    def edit_file_anthropic(self, file_path, problem_statement):
        # Read the original content of the file
        with open(self.local_repo_path / file_path.strip("/"), 'r') as file:
            original_content = file.read()

        # Initialize the prompt with the problem statement and the original content of the file
        prompt = f"Given the problem statement:\n\n{problem_statement}\n\nAnd the content of the file {file_path}:\n\n```python\n{original_content}\n```\n\n"
        prompt += "Please edit the file to solve the problem and output the edited file content. Respond only with code in a single Markdown code block (starting with ```python) with no explanation because your response will be pasted into the file."
        print(f"Prompt for {file_path}:")
        # print(prompt)

        # Ensure the API key is set
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")

        # Initialize the ChatAnthropic API
        chat = ChatAnthropic(anthropic_api_key=anthropic_api_key, model='claude-3-opus-20240229', temperature=0)

        # Send the prompt to Claude and get the response
        response = chat.invoke(prompt)

        # Splitting the response content to extract code
        parts = response.content.split("```python", 1)
        print("PARTS:", parts)

        if len(parts) > 1:
            # Remove all instances of "```" from the edited content
            edited_content = parts[1].replace("```", "").strip()

            print(f"Edited content for {file_path}:\n{edited_content}")

            # Overwrite the file with the edited content
            with open(self.local_repo_path / file_path.strip("/"), 'w') as file:
                file.write(edited_content)
        else:
            print(response.content)
            raise Exception("No code block found in Claude's response. Stopping execution.")


    def generate_patches(self):
        problem_statement = self.dataset.get("problem_statement", "")
        instance_id = self.dataset["instance_id"]
        model_name_or_path = "seven-v1"  # Adjust as necessary

        nearest_files_contents = self.get_nearest_files(problem_statement)
        print(f"Found {len(nearest_files_contents)} nearest files.")

        # Create a temporary branch for edits
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        branch_name = f'temp_branch_for_edits_{timestamp}'
        repo = Repo(str(self.local_repo_path))
        repo.git.checkout(self.base_commit, b=branch_name)

        # Apply edits and generate diffs
        for file_path, _ in nearest_files_contents:
            self.edit_file(file_path, problem_statement)
        diff = repo.git.diff(self.base_commit)

        # Checkout back and delete the temporary branch
        # repo.git.checkout('main')
        # repo.git.branch('-D', branch_name)

        # Construct the prediction entry
        prediction_entry = {
            "instance_id": instance_id,
            "model_patch": diff,
            "model_name_or_path": model_name_or_path,
        }

        # Path to your predictions file
        predictions_path = self.instance_path / "predictions.json"

        # Load existing predictions if file exists, else start with an empty list
        if predictions_path.exists():
            with open(predictions_path, 'r') as file:
                predictions = json.load(file)
        else:
            predictions = []

        # Append the new prediction and save
        predictions.append(prediction_entry)
        with open(predictions_path, 'w') as file:
            json.dump(predictions, file, indent=4)

        print(f"Patch saved for {instance_id} in {predictions_path}")


    def generate_patches_claude(self):
        problem_statement = self.dataset.get("problem_statement", "")
        nearest_files_contents = self.get_nearest_files(problem_statement)
        print(f"Found {len(nearest_files_contents)} nearest files.")

        # Ensure paths are relative to local_repo_path by removing any leading slash, then combine with local_repo_path
        fnames = [self.local_repo_path / file_path.lstrip("/") for file_path, _ in nearest_files_contents]

        # Ensure the paths are converted to strings if needed by the API or further processing
        fnames_str = [str(fname) for fname in fnames]

        # Initialize the prompt with the problem statement
        prompt = f"Given the problem statement:\n\n{problem_statement}\n\n"

        # Append the content of each nearest file to the prompt
        for file_path, content in nearest_files_contents:
            # Ensure the file path is a string relative to the local_repo_path, stripping any leading slash
            file_path_str = str(self.local_repo_path / file_path.lstrip("/"))
            prompt += f"Contents of {file_path_str}:\n\n```python\n{content}\n```\n\n"

        prompt += "Please provide a code patch that solves the problem."

        # Ensure the API key is set
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")

        chat = ChatAnthropic(anthropic_api_key=anthropic_api_key, model='claude-3-opus-20240229', temperature=0)

        # Adjust the call to invoke with a string prompt directly
        response = chat.invoke(prompt)

        # Print the response from the model
        print("Claude's response:", response.content)


    def get_nearest_files(self, query, num_hits=1):
        print(Fore.BLUE + "Checking for nearest files to query, num_hits:", num_hits)

        # Ensure there are embeddings to compare against
        if not hasattr(self, 'embeddings') or self.embeddings.size == 0:
            print(Fore.RED + "No embeddings available to perform the search.")
            return []

        # Get the embedding for the query
        query_embedding = embed([query])[0]

        # Normalize if your file embeddings were normalized
        query_embedding /= np.linalg.norm(query_embedding)

        # Calculate the cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Sort the files by decreasing similarity
        nearest_indices = np.argsort(similarities)[::-1][:num_hits]

        # Load the descriptions to find the corresponding file paths
        descriptions = self.load_descriptions()

        # Map indices back to file paths using the descriptions
        paths = list(descriptions.keys())  # Assuming the descriptions' keys are the relative file paths
        nearest_file_paths = [paths[index] for index in nearest_indices if index < len(paths)]

        nearest_files_contents = []
        for file_path in nearest_file_paths:
            try:
                with open(self.local_repo_path / file_path.strip("/"), 'r') as file:
                    nearest_files_contents.append((file_path, file.read()))
            except Exception as e:
                print(Fore.RED + f"Error accessing file {file_path}: {e}")

        return nearest_files_contents
