#!/usr/bin/env python3
"""Fork selected repos to seanlab007 and create Whale1.0 project structure."""
import subprocess
import requests
import json
import time
import os
import sys

def get_github_token():
    result = subprocess.run(
        ["security", "find-internet-password", "-s", "github.com", "-a", "seanlab007", "-w"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Repos to fork (owner/repo format)
REPOS_TO_FORK = [
    ("Wan-Video", "Wan2.2"),
    ("THUDM", "CogVideo"),
    ("Tencent-Hunyuan", "HunyuanVideo"),
    ("hpcaitech", "Open-Sora"),
    ("genmoai", "mochi"),
    ("Lightricks", "LTX-Video"),
    ("SkyworkAI", "SkyReels-V1"),
    ("KwaiVGI", "LivePortrait"),
    ("Rudrabha", "Wav2Lip"),
    ("guoyww", "AnimateDiff"),
    ("megvii-research", "ECCV2022-RIFE"),
]

token = get_github_token()
headers = {"Authorization": f"Bearer {token}"}
BASE = "https://api.github.com"

# Check existing repos first
search_resp = requests.get(
    f"{BASE}/search/repositories?q=fork:true+user:seanlab007",
    headers=headers
)
existing = set()
if search_resp.status_code == 200:
    for item in search_resp.json().get("items", []):
        existing.add(item["name"])
    print(f"Existing forked repos ({len(existing)}): {sorted(existing)}")

# Verify authentication
user_resp = requests.get(f"{BASE}/user", headers=headers)
if user_resp.status_code == 200:
    print(f"Authenticated as: {user_resp.json()['login']}")
else:
    print(f"Authentication failed: {user_resp.status_code}")
    sys.exit(1)

# Fork each repo
results = []
for owner, repo in REPOS_TO_FORK:
    if repo in existing:
        print(f"\n[{repo}] Already forked, skipping")
        results.append({"repo": repo, "status": "already_exists"})
        continue
    
    print(f"\n[{repo}] Forking {owner}/{repo}...")
    url = f"{BASE}/repos/{owner}/{repo}/forks"
    resp = requests.post(url, headers=headers, json={"default_branch_only": True})
    
    if resp.status_code in (200, 201, 202):
        data = resp.json()
        print(f"  -> Forked! Clone URL: {data.get('clone_url', 'N/A')}")
        print(f"  -> HTML URL: {data.get('html_url', 'N/A')}")
        results.append({
            "repo": repo,
            "status": "forked",
            "clone_url": data.get("clone_url", ""),
            "html_url": data.get("html_url", "")
        })
        time.sleep(1)  # Rate limit
    elif resp.status_code == 202:
        print(f"  -> Accepted (async fork), will check status")
        results.append({"repo": repo, "status": "accepted"})
    else:
        err = resp.text[:200]
        print(f"  -> FAILED: {resp.status_code} {err}")
        results.append({"repo": repo, "status": "failed", "error": err})

# Summary
print("\n\n===== FORK SUMMARY =====")
for r in results:
    icon = "✅" if r["status"] in ("forked", "already_exists") else "❌"
    print(f"  {icon} {r['repo']}: {r['status']}")

print("\nDone!")
