"""Require a completed successful CI run for the exact production candidate."""
from __future__ import annotations

import json
import os
import re
from urllib.request import Request, urlopen


def qualified_run(runs: list[dict], sha: str) -> dict:
    candidates = [run for run in runs if run.get("head_sha") == sha and run.get("head_branch") == "main"
                  and run.get("event") in {"push", "workflow_dispatch"}]
    if not candidates:
        raise ValueError("Exact candidate has no main CI run")
    latest = max(candidates, key=lambda run: (run["run_number"], run.get("run_attempt", 1)))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError("Latest CI attempt for this candidate did not complete successfully")
    return latest


def main() -> None:
    repo, sha = os.environ["GITHUB_REPOSITORY"], os.environ["RELEASE_SHA"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Invalid repository/candidate identity")

    def get(path):
        request = Request("https://api.github.com/repos/" + repo + path,
                          headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                                   "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    if get("/commits/main")["sha"] != sha:
        raise ValueError("Candidate is no longer remote main; refusing stale production deploy")
    runs = get("/actions/workflows/ci.yml/runs?branch=main&head_sha=" + sha + "&per_page=100")["workflow_runs"]
    run = qualified_run(runs, sha)
    print(json.dumps({"qualified_sha": sha, "ci_run_id": run["id"], "attempt": run.get("run_attempt", 1)}))


if __name__ == "__main__":
    main()
