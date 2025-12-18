import os
import yaml
import subprocess
import sys 

# Load Config
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

def run_command(command, cwd=None):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        # Don't exit immediately, some git commands (like commit with no changes) might fail harmlessly
        return False
    return True

def main():
    hugo_bin = CONFIG['paths'].get('hugo_bin', 'hugo')
    public_dir = CONFIG['paths'].get('public_dir', 'public')
    source_branch = CONFIG['publisher'].get('source_branch', 'main')
    
    # 1. Save Source (Git Add/Commit/Push)
    print(">>> saving source code...")
    run_command("git add .")
    run_command('git commit -m "Auto-save: Research progress"')
    run_command(f"git push origin {source_branch}")

    # 2. Build Site
    print(">>> Building Hugo site...")
    # Check if executable exists or use system path
    if os.path.exists(hugo_bin):
        hugo_bin = os.path.normpath(hugo_bin)
        build_cmd = f"{hugo_bin} --destination {public_dir}"
    else:
        build_cmd = f"hugo --destination {public_dir}"
        
    if not run_command(build_cmd):
        print("Build failed.")
        sys.exit(1)

    # 3. Deploy to gh-pages
    print(">>> Deploying to gh-pages...")
    # ghp-import -n (no-jekyll) -p (push) -f (force) -b (branch)
    deploy_cmd = f"venv\\Scripts\\ghp-import -n -p -f {public_dir}"
    
    if run_command(deploy_cmd):
        print(">>> Published Successfully!")
    else:
        print(">>> Deployment Failed.")

if __name__ == "__main__":
    main()
