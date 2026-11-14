---
name: Local dreamlite dir is not a git repo
description: The /data/data/com.termux/files/home/projects/dreamlite directory must never be treated as a git repo or have its contents committed/pushed anywhere
type: project
---

The user explicitly stated the local `dreamlite` working directory is NOT a git repository and forbade adding anything from it into any git repository.

**Why:** The directory is a loose working/scratch area; the real GitHub repo (`maoist2009/qwen35`) is separate and owned by the user. They do not want me to init a git repo here, commit, or push these files.

**How to apply:** Never run `git init`/`git add`/`git commit`/`git push` against this directory, and do not upload its files to any remote (including via the GitHub Contents API) without explicit user authorization. GitHub Actions runs live on the separate `maoist2009/qwen35` repo, which the user manages.
