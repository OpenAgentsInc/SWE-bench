from colorama import Fore, Style, init
from harness_devin.scripts import TestSpec # Class with fields: instance_id, setup_script, prompt, eval_script
from harness_devin.types import SwebenchInstance # Dict with keys: repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit

init(autoreset=True)

def run(dataset: SwebenchInstance, test_spec: TestSpec):
    print(Fore.MAGENTA + 'Seven will assimilate ' + dataset["instance_id"])
