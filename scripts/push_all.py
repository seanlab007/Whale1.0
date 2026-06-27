#!/usr/bin/env python3
"""Push Whale1.0 to all 4 platforms: GitHub, GitLab, Gitee, local backup."""
import subprocess, json, os, time, sys

def get_token(service, account):
    r = subprocess.run(
        ["security", "find-internet-password", "-s", f"{service}.com", "-a", account, "-w"],
        capture_output=True, text=True
    )
    return r.stdout.strip()

def create_github_repo(token, name, description):
    import requests
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    # Check if exists
    r = requests.get(f"https://api.github.com/repos/seanlab007/{name}", headers=headers)
    if r.status_code == 200:
        print(f"[GitHub] Repo already exists: seanlab007/{name}")
        return True
    # Create
    data = {
        "name": name,
        "description": description,
        "private": False,
        "auto_init": False
    }
    r = requests.post("https://api.github.com/user/repos", headers=headers, json=data)
    if r.status_code in (200, 201):
        print(f"[GitHub] Created: seanlab007/{name}")
        return True
    else:
        print(f"[GitHub] Failed: {r.status_code} {r.text[:200]}")
        return False

def create_gitlab_repo(token, name, description):
    import requests
    headers = {"PRIVATE-TOKEN": token}
    r = requests.post("https://gitlab.com/api/v4/projects", headers=headers, json={
        "name": name, "description": description, "visibility": "public"
    })
    if r.status_code in (200, 201):
        print(f"[GitLab] Created: seanlab007/{name}")
        return r.json().get("http_url_to_repo", "")
    elif r.status_code == 409:
        print(f"[GitLab] Already exists")
        return f"https://gitlab.com/seanlab007/{name}.git"
    else:
        print(f"[GitLab] Failed: {r.status_code} {r.text[:200]}")
        return None

def create_gitee_repo(token, name, description):
    import requests
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    r = requests.post(f"https://gitee.com/api/v5/user/repos?access_token={token}", 
                      headers=headers, json={
        "name": name, "description": description, "visibility": 0, "auto_init": False
    })
    if r.status_code in (200, 201):
        print(f"[Gitee] Created: seanlab007/{name}")
        return r.json().get("clone_url", "")
    elif r.status_code == 409:
        print(f"[Gitee] Already exists")
        return f"https://gitee.com/seanlab007/{name}.git"
    else:
        print(f"[Gitee] Failed: {r.status_code} {r.text[:200]}")
        return None

# Config
PROJECT_DIR = "/Users/mac/WorkBuddy/2026-06-28-00-29-42/whale1.0"
REPO_NAME = "Whale1.0"
DESCRIPTION = "🐋 Whale1.0 - Open Source Video Generation Foundation Model. Integrated top-tier models: Wan2.2, CogVideoX, HunyuanVideo, Mochi, LTX-Video."

print("=" * 60)
print("🐋 Syncing Whale1.0 to 4 platforms")
print("=" * 60)

# Get tokens
github_token = get_token("github", "seanlab007")
gitlab_token = get_token("gitlab", "seanlab007")
gitee_token = get_token("gitee", "seanlab007")

# Step 1: Create repos on all platforms
print("\n📦 Step 1: Creating remote repositories...")
create_github_repo(github_token, REPO_NAME, DESCRIPTION)
time.sleep(1)
gitlab_url = create_gitlab_repo(gitlab_token, REPO_NAME, DESCRIPTION)
time.sleep(1)
gitee_url = create_gitee_repo(gitee_token, REPO_NAME, DESCRIPTION)

# Step 2: Add remotes and push
print("\n📤 Step 2: Adding remotes and pushing...")
os.chdir(PROJECT_DIR)

# GitHub
subprocess.run(["git", "remote", "add", "github", f"https://seanlab007:{github_token}@github.com/seanlab007/{REPO_NAME}.git"], 
               capture_output=True)
print("[Remote] Added: github")

# GitLab  
if gitlab_url:
    subprocess.run(["git", "remote", "add", "gitlab", gitlab_url.replace("https://", f"https://seanlab007:{gitlab_token}@")],
                   capture_output=True)
    print("[Remote] Added: gitlab")

# Gitee
if gitee_url:
    subprocess.run(["git", "remote", "add", "gitee", gitee_url.replace("https://", f"https://seanlab007:{gitee_token}@")],
                   capture_output=True)
    print("[Remote] Added: gitee")

# Push to all
remotes = subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip().split()
print(f"\nGit remotes: {remotes}")

for remote in remotes:
    print(f"\n  Pushing to {remote}...")
    result = subprocess.run(
        ["git", "push", "-u", remote, "main", "--force"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        print(f"  ✅ {remote}: Pushed successfully!")
    else:
        # Try with default branch
        stderr = result.stderr.lower()
        if "main" in stderr and "does not exist" in stderr:
            result2 = subprocess.run(
                ["git", "push", "-u", remote, "main", "--force"],
                capture_output=True, text=True, timeout=120
            )
            if result2.returncode == 0:
                print(f"  ✅ {remote}: Pushed successfully!")
            else:
                print(f"  ❌ {remote}: {result2.stderr[:200]}")
        else:
            print(f"  ❌ {remote}: {result.stderr[:200]}")

print("\n" + "=" * 60)
print("🎉 Sync complete! Check the repos:")
print(f"   GitHub: https://github.com/seanlab007/{REPO_NAME}")
print(f"   GitLab: https://gitlab.com/seanlab007/{REPO_NAME}")
print(f"   Gitee:  https://gitee.com/seanlab007/{REPO_NAME}")
print(f"   Local:  {PROJECT_DIR}")
print("=" * 60)
