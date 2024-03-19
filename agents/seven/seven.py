import os
from github import Github

from colorama import Fore, Style, init

class Seven:
    def __init__(self, repo):
        self.token = os.getenv('GITHUB_TOKEN')
        self.github = Github(self.token)
        self.user = self.github.get_user()
        self.upstream_repo = self.github.get_repo(repo)
        self.fork_repo(self.upstream_repo)
        name = repo.split('/')[1]
        self.repo = self.user.get_repo(name)
        print(Fore.GREEN + f"Repository {self.repo} created.")

    def fork_repo(self, repo):
        if repo.name not in [r.name for r in self.user.get_repos()]:
            self.user.create_fork(repo)
