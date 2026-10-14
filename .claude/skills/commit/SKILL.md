---
name: commit
description: Commit the current changes with a short, summary-level message. Use when the user asks to commit, or runs /commit.
---

# Commit

1. Run `git status` and `git diff` (plus `git diff --staged`) to see what changed.
2. Decide what to commit:
   - If files are already staged, commit only those.
   - Otherwise stage the files that belong to the work just done. Leave out unrelated or generated changes (e.g. `metrics.json` from a training run, caches) unless the user asked for them. List anything left out in your reply.
   - If the changes are clearly unrelated to each other (e.g. a training fix and a README edit), make one commit for each.
3. Write the message:
   - **One line, no body.** Lowercase, under ~70 characters.
   - Start with the repo's prefix: `feat:`, `fix:`, `docs:`, `refactor:`, or `chore:`.
   - Say what changed at a summary level, naming the main file or area when helpful. Don't explain why, list details, or describe how.
   - Good: `fix: update index.py to handle ArcFace weights better`
   - Good: `feat: add wiki_search.py for free image search`
   - Good: `docs: add training and web setup to readme`
   - Too long: `fix: make classifier head optional so checkpoints resume after class count changes, and filter old keys in api.py`
4. If the session requires an attribution trailer (e.g. `Co-Authored-By:`), add it after a blank line. It doesn't count as a body.
5. Commit, then reply with the commit hash and message. Don't push unless asked.
