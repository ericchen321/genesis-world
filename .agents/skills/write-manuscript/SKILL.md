---
name: write-manuscript
description: Compose a new LaTeX manuscript or edit an existing manuscript in a project folder whose file structure already exists. Use this when the user provides a manuscript/project path and asks Codex to fill in or revise paper text, not figures.
---

# write-a-manuscript

## Purpose

Use this skill to write, revise, or complete a manuscript organized as a project folder. The user will provide the project path. Assume the folder structure, LaTeX files, figure files, bibliography files, and supporting project materials already exist. Your job is to fill in or edit textual manuscript content and compile the manuscript.

Do not create or modify figures unless the user explicitly asks. Figure files are assumed to be supplied by the user.

## Required inputs

Before starting, identify:

- `PROJECT_PATH`: the root folder of the manuscript project.
- `USER_GOAL`: whether the task is to compose a new manuscript, complete missing sections, or edit an existing manuscript.
- `TARGET_SCOPE`: whole paper, specific sections, specific `.tex` files, rebuttal-related edits, camera-ready polish, etc.
- `VENUE_OR_STYLE`, if supplied by the user.
- Any user-provided constraints about framing, claims, results, citations, tone, length, or sections to avoid touching.

If `PROJECT_PATH` is missing, ask for it once. If the path is provided, do not ask for confirmation before inspecting the project.

## Non-negotiable workflow

Always use the following four sub-agents:

1. **Explorer** — GPT-5.4 mini, medium reasoning.
2. **Planner** — latest full model, high reasoning.
3. **Writer** — latest full model, medium reasoning.
4. **Validator** — GPT-5.4 mini, medium reasoning.

Use Codex's sub-agent/task mechanism if available. Spawn the explorer(s) first to gather information, then the planner to design the editing strategy, then the writer to implement the edits, and finally, close the explorers and the planner, leave the writer open, and spawn the validator to check the work. After the first Writer pass, run the Validator and follow its guidance for any necessary revisions.

The Writer and Validator must run a write-validate-write cycle until the Validator believes the plan has been successfully implemented. If the loop cannot converge because required information is missing, stop only after clearly listing the blocker, the affected sections, and the exact missing evidence or reference. Close all agents when finished and report the final status to the user.

## Global rules

- Treat the project folder as the source of truth.
- Preserve the existing manuscript structure unless the Planner explicitly recommends moving text across files.
- Edit text, equations, captions, sectioning, labels, references, and LaTeX macros only when needed for the manuscript task.
- Do not edit figure image files.
- Do not invent experimental results, ablations, metrics, dates, numbers, datasets, baselines, or implementation details.
- Do not invent references.
- Do not add a citation unless the cited work exists in the project bibliography or the user supplied the reference material.
- If a claim needs a citation and no valid reference exists, write a precise `TODO: citation needed for ...` note rather than fabricating a BibTeX entry. Highlight the TODO text in red.
- Keep terminology, notation, and claims consistent across abstract, introduction, method, experiments, conclusion, and captions.
- Prefer targeted edits over unnecessary rewrites.
- Compile after edits and fix LaTeX errors introduced by the edits.
- Before finishing, check for unresolved placeholders, citation errors, undefined references, and inconsistent claims.

## Project discovery

From `PROJECT_PATH`, inspect at least:

- Top-level files: `README*`, `Makefile`, `latexmkrc`, `.gitignore`, build scripts, submission instructions.
- LaTeX sources: `*.tex`, `sections/*.tex`, `main.tex`, `paper.tex`, `appendix.tex`, `supplement.tex`.
- Bibliography files: `*.bib`, venue style files, class files.
- Figure references: `figures/`, `imgs/`, `assets/`, or any paths used by `\includegraphics`.
- Project context: notes, work logs, experiment logs, result summaries, code comments, configs, scripts, tables, captions, and user-provided prompt files.
- Existing PDF, if present, to understand the current manuscript state.

Avoid spending time on large generated artifacts, raw datasets, checkpoints, videos, or binary files unless they are directly relevant to claims in the manuscript.

Useful commands:

```bash
cd "$PROJECT_PATH"
find . -maxdepth 3 -type f | sed 's#^./##' | sort | head -300
rg -n "TODO|TBD|XXX|PLACEHOLDER|\\todo|\\cite\{|\\ref\{|\\includegraphics|begin\{figure\}|begin\{table\}" .
find . -maxdepth 3 -type f \( -name '*.tex' -o -name '*.bib' -o -name 'README*' -o -name '*.md' \) -print
```

## Agent 1: Explorer

**Model:** GPT-5.4 mini
**Reasoning:** medium

The Explorer gathers context only. It should not edit files. Generally one explorer subagent is sufficient, but a user should be allowed to explicitly ask for multiple explorers, each tasked with a different focus area.

Explorer responsibilities:

1. Map the manuscript structure.
2. Identify the main LaTeX entry point and compile command, if obvious.
3. Summarize each relevant `.tex` file and its current completeness.
4. Gather necessary context from:
   - user prompt,
   - project notes,
   - work logs,
   - codebase comments or READMEs,
   - experiment/result files,
   - existing manuscript text,
   - bibliography.
5. Identify available figures and what claims they can support.
6. Identify all existing bibliography entries relevant to the task.
7. Flag missing information, weak evidence, or places where the manuscript currently overclaims.
8. Produce an `Explorer Brief` for the Planner.

Each explorer shall provide the following information to the Planner:
- User goal
- Project structure (parts the agent has explored)
- Main LaTeX entry points
- Existing manuscript status
- Key factual context from project materials
- Available figures/tables and supported claims (if asked to explore)
- Available bibliography entries (if asked to explore)
- Missing evidence / unresolved questions
- Relation of this project to prior work (if relevant and identifiable from project materials)

## Agent 2: Planner

**Model:** latest full model
**Reasoning:** high

The Planner designs the manuscript-editing strategy from the Explorer Brief. It should not edit files.

Planner responsibilities:

1. Decide the paper story, framing, and contribution claims consistent with available evidence.
2. Decide which `.tex` files need edits and what each edit should accomplish.
3. Specify section-level content plans.
4. Specify where to use existing figures and tables, without asking the Writer to create new figures.
5. Specify which citations may be used, and only from available bibliography entries.
6. Identify claims that must be marked as TODO due to missing evidence. For claims to be supported by an experiment which has no results uploaded yet, mark the claim as `TO VALIDATE: ...` and highlight it with red text. Also, leave placeholder for user to fill in the results when they are available, e.g. `TODO: add results for <experiment>`.
7. Define validation criteria for the Validator.
8. Produce an editing plan detailed enough for the Writer to implement without guessing.

Planner output format:

```markdown
# Edit Plan

## Manuscript objective

## Core story and contribution claims

## File-by-file edit plan

### <file path>
- Sections to edit:
- Text to add/change/remove:
- Claims planned, and each claim's current state (supported, support-in-progress);
- Bib citations to be added, with allowed BibTeX keys:
- Figures/tables to reference:
- Consistency requirements:

## Claims that must not be made

## TODOs allowed if evidence is missing

## Compile command

## Validator checklist
```

Save the edit plan under `.codex/plans/EDIT_<come up with a name>.md` for the Writer to access.

## Agent 3: Writer

**Model:** GPT-5.5
**Reasoning:** medium

The Writer implements the Planner's editing plan in the project files. Before commencing the writing, the writer should digest the plan as well as information from the explorers.

Writer responsibilities:

1. Digest the Planner's editing plan and information from the explorers before commencing writing.
2. Edit only the files named or implied by the plan.
3. Follow the plan closely.
4. Maintain existing LaTeX style, macros, notation, labels, and section organization.
5. Use citations only from the allowed citation list or existing `.bib` entries explicitly approved by the Planner.
6. Do not fabricate references or BibTeX entries.
7. Do not fabricate results, numbers, dataset details, or baselines. When results are not available yet to support a claim, add a placeholder command (`\placeholder{...}`) with instructions for the user to fill in when the results are ready, and mark the claim as `TO VALIDATE: ...` with red text.
8. Add clear TODOs for missing but necessary evidence rather than inventing content.
9. Compile the manuscript using the planned build command.
10. Fix compile errors caused by the edits.
11. Produce a concise `Writer Report` summarizing changed files, compile status, and known TODOs.

Writer shall directly edit the project files. After the write-up completes and before validator checking, the main agent shall close all explorer agents and the planner agent to free up resources.

Writer shall always write, with confidence, in a voice as if all final results and evidence are in place, even if they are not. The Writer should not write in a tentative or hypothetical tone. Instead, the Writer should inject precise placeholders and TODOs to indicate (to human user) where evidence is missing, without weakening the language of the supported claims. The Writer is also encouraged to use active voice whenever it sees fit, e.g. "we show that"; do not proliferate passive voice, e.g. "it can be shown that".

## Agent 4: Validator

**Model:** GPT-5.4 mini
**Reasoning:** medium

The Validator checks the completed or modified manuscript against the Planner's planning document. It should inspect the changed `.tex` files, bibliography, and compile log/PDF if available.

Validator responsibilities:

1. Confirm every required point in the planning document has been addressed.
2. Check cross-section consistency:
   - abstract vs introduction,
   - contributions vs method,
   - method vs experiments,
   - experiments vs conclusion,
   - captions vs main text,
   - supplement vs main paper, if applicable.
3. Check that no claims "come out of nowhere", i.e. not claimed by human input or supported by existing evidence (e.g. referenced bib entries), and not even marked as TODO/TO-VALIDATE.
4. Check that there are no made-up references.
5. Verify every citation key used in changed text exists in a `.bib` file.
6. Verify that newly introduced citations correspond to the correct cited work, based on the bibliography title/authors and project context.
7. Check for undefined references, broken labels, empty citations, duplicated labels, and suspicious placeholders.
8. Check the compile log for errors and important warnings.
9. Decide whether another Writer pass is required.
10. Compilation: e.g. compile from `corl26_manuscript/corl_2026_template_cameraready/` using command `latexmk -pdf -bibtex <name of the main .tex file>`. Make sure no errors are present in the log. If errors are present, report them clearly and do not proceed to final validation until they are resolved.

Suggested validation commands:

```bash
cd "$PROJECT_PATH"
rg -n "TODO|TBD|XXX|PLACEHOLDER|citation needed|\\cite\{\}|\\ref\{\}|undefined|Undefined|multiply defined|LaTeX Warning|Citation.*undefined" .
python - <<'PY'
from pathlib import Path
import re
tex = '\n'.join(p.read_text(errors='ignore') for p in Path('.').rglob('*.tex'))
bib = '\n'.join(p.read_text(errors='ignore') for p in Path('.').rglob('*.bib'))
keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib))
used = set()
for m in re.finditer(r'\\(?:cite|citet|citep|citealp|citeauthor|citeyear|autocite|parencite|textcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        k = k.strip()
        if k:
            used.add(k)
missing = sorted(used - keys)
print('Missing citation keys:', missing)
PY
```

Validator output format:

```markdown
# Validation Report

## Pass/fail decision

Use one of:
- PASS: the plan is implemented successfully.
- REVISE: another Writer pass is required.
- BLOCKED: missing evidence or user input prevents completion.

## Plan coverage

## Logic and consistency issues

## Reference and citation audit

## Compile/log audit

## Required next Writer actions
```

## Write-validate-write loop

After the first Writer pass:

1. Run the Validator.
2. If Validator returns `PASS`, proceed to final reporting.
3. If Validator returns `REVISE`, send the `Validation Report` back to the Writer.
4. Writer performs only the required next actions.
5. Compile again.
6. Validator checks again.
7. Repeat until Validator returns `PASS`.
8. If Validator returns `BLOCKED`, do not invent missing content. Report the blocker clearly.

The loop should converge by fixing concrete issues, not by weakening the validation standard.

## Reference integrity protocol

This is the most important quality gate.

Before finalizing:

- Extract all citation keys from changed `.tex` files.
- Confirm every key exists in a `.bib` file.
- Confirm every newly used key matches the intended cited work.
- Do not rely on memory for paper titles, authors, years, venues, or claims.
- Do not add BibTeX entries from memory.
- If the user asks to add a new reference, use only user-supplied BibTeX or a verified source already present in the project materials.
- If a needed reference is unavailable, leave `TODO: add verified citation for <claim>`.

Never fabricate references to make the manuscript look complete.

## Claim integrity protocol

For every substantive claim, ensure it is supported by at least one of:

- existing manuscript content,
- project notes or logs,
- experiment/result files,
- code/config files,
- user-supplied prompt,
- cited literature already present in the bibliography.

If unsupported, mark it with `TO VALIDATE: ...` and highlight it with red text. If experiment results are needed for supporting the claim (check the Experiment section if an experiment is planned to support the claim), provide user with a placeholder highlighted in red. Be especially strict with:

- quantitative improvements,
- baseline comparisons,
- dataset size,
- user study details,
- real-robot results,
- runtime/performance claims,
- claims of novelty or first-of-kind status,
- statements about what prior work does or does not do.

## Compilation protocol

Prefer the project's own compile command. Use the first applicable option:

1. Command documented in `README`, `Makefile`, `latexmkrc`, or submission notes.
2. `make`, if a clear manuscript target exists.
3. `latexmk -pdf -interaction=nonstopmode -halt-on-error <main>.tex`.
4. If the project requires BibTeX/Biber manually, run the appropriate sequence and document it.

After compiling:

- Inspect the log.
- Fix errors introduced by edits.
- Do not delete content merely to make compilation pass unless the deletion is part of the plan or clearly necessary.
- If compilation fails because of pre-existing missing figures, missing style files, or unavailable external assets, report that separately from manuscript-text issues.

## Final response to user

When finished, report:

- whether the task passed validation,
- files changed,
- compile result,
- major content added or revised,
- remaining TODOs or blockers,
- any reference/citation issues found and how they were handled.

Keep the final response concise and actionable.
