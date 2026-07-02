---
name: distill-and-retrieve-branch-reusable-knowledge
description: Maintain and consult a branch-specific reusable knowledge base for this repository. Use it to record durable findings that may help future threads on the same Git branch, while treating all entries as potentially stale after code changes. Invoke this skill 1) to distill knowledge after a main or a sub agent has been invoked to explore the repository and obtained finding; 2) to retrieve knowledge at the start of a new task on the same branch, during the phase when code exploration takes place.
---

# Branch Reusable Knowledge

## Purpose

This skill maintains a **branch-specific reusable knowledge base** inside the repository.

Its goals are:

1. Record findings that are likely to be useful again in later threads on the **same Git branch**.
2. Reuse prior knowledge at the start of a new task, before re-discovering the same information.
3. Treat all stored knowledge as **tentative** and potentially **outdated**, because code may have changed since the note was written.

This is **not** a source of truth. It is a working memory aid.

---

## When to use this skill

Use this skill whenever upon exploring this repository, the agent:
- discovers something that will likely be useful again,
- learns repository-specific conventions,
- identifies recurring build/test/debug facts,
- traces relationships between components that are expensive to rediscover.

Do **not** use this skill for trivial, one-off observations with no future value.

---

## Storage location

Store knowledge under:

`./.agents/knowledge/`

Within that directory, maintain:

- one index file:
  - `./.agents/knowledge/README.md`
- one branch-specific file per Git branch:
  - `./.agents/knowledge/<branch-name>.md`

If needed, sanitize branch names so they are valid filenames, for example:

- `/` -> `__`
- spaces -> `_`

Example:

- branch `main` -> `./.agents/knowledge/main.md`
- branch `feature/sim-refactor` -> `./.agents/knowledge/feature__sim-refactor.md`

---

## Core behavior

### 1. At the beginning of a task, check for reusable knowledge

Before doing substantial work:

1. Determine the current Git branch.
2. Look for the branch-specific knowledge file.
3. If it exists, read it first.
4. Use any relevant notes as hints, not facts.
5. Verify important or risky claims against the current code before relying on them.

If the file does not exist, continue normally.

---

### 2. While working, record reusable findings

When you discover information that is likely to help future threads on the same branch, add it to the branch knowledge file.

Good candidates include:

- architecture facts specific to this branch,
- locations of key entry points,
- non-obvious module ownership,
- build/test commands that actually work,
- common failure modes and their causes,
- naming conventions,
- migration/refactor status,
- temporary branch-specific hacks or known limitations,
- task-relevant assumptions that future threads should re-check.

Do not record:

- secrets,
- credentials,
- tokens,
- personal data,
- large copied code blocks,
- generic advice not specific to this repo/branch,
- ephemeral scratch notes that only matter for the current thread.

---

### 3. Always mark knowledge as uncertain over time

Because the repository evolves, every knowledge entry must be treated as possibly stale.

For each entry, include:

- date,
- branch,
- confidence level,
- evidence,
- staleness warning.

Prefer wording like:

- “As of this inspection...”
- “Likely true on this branch at the time of writing...”
- “Needs re-verification if related files changed...”

Never present the knowledge base as guaranteed correct.

---

## File format

Use Markdown.

Each branch file should start with this structure:

```md
# Branch Knowledge: <branch-name>

> This file stores reusable knowledge for the Git branch `<branch-name>`.
> All entries are tentative and may become outdated after code changes.
> Treat this file as a hint source, not a source of truth.

## How to use
- Read this file at the start of a new task on this branch.
- Re-verify important claims against the current code.
- Add durable findings that future threads can reuse.

## Knowledge Entries