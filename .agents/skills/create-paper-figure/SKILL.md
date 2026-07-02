---
name: create-paper-figure
description: Create an architecture diagram for a manuscript submission using draw.io. Takes a hand-drawn layout, a reference architecture figure for style, and any other user-provided resources. Spawns an explorer, planner, element-generation, and drawer subagent; iterates a draw-review-draw cycle until the planner approves the figure. Saves the .drawio source and exported PNG under a user-specified directory.
---

# create-paper-figure

Produce a polished architecture diagram for a conference paper submission using the `drawio-skill`, driven by a multi-agent pipeline with an explicit draw-review-draw loop.

---

## Overview

| Actor | Model | Role | Active when |
|---|---|---|---|
| **Explorer** | GPT-5.4 mini, high reasoning | Digests all user-provided resources; surfaces the information most relevant to this figure | Before planning begins |
| **Planner** | GPT-5.5, x-high reasoning | Designs the figure layout and content; reviews the produced figure; directs fixes | After Explorer finishes; during each review round |
| **Element generator** | GPT-5.5, high reasoning | Generates stand-alone icons and visual elements (with GPT Image 2, via @Image Gen command); saves them to the output directory | After Planner issues its first directive |
| **Drawer** | GPT-5.5, high reasoning | Writes and iteratively edits the draw.io XML; exports draft and final PNG | After Planner issues its first directive; during each fix round |
| **Main agent** | — | Orchestrator only — collects inputs, routes context between agents, enforces the loop | Throughout |

The main agent's **sole duty** is coordination. It must not deviate from this workflow, interpret design decisions on behalf of the Planner, or shortcut any phase.

---

## Phase 0 — Gather inputs

Before spawning any subagent, the main agent must collect the following from the user. Ask for all three in a single message:

1. **Crude layout** — a hand-drawn or rough sketch of the figure's intended layout (image or annotated file). This defines the spatial structure to follow.
2. **Reference architecture figure** — an architecture diagram from another project whose color palette and icon style should be adopted.
3. **Output directory** — the path under which the `.drawio` source file, exported PNG, and any generated icon files should be saved. Create it with `mkdir -p` if it does not exist.
4. **(Optional) Additional context** — any other files, directories, paper drafts, or notes the user wants the agents to consult (e.g., manuscript section describing the architecture, existing figure assets).

Do not proceed until items 1–3 are provided.

---

## Phase 1 — Explore

Spawn the **Explorer** subagent with all user-provided resources as input.

### Explorer objectives
- Read and synthesize the crude layout, the reference architecture figure, the manuscript/paper context (if any), and any other provided files.
- Identify the components, data flows, and relationships that must appear in the figure.
- Extract the color palette, icon style, shape conventions, and typography rules from the reference architecture figure.
- Note layout constraints visible in the crude sketch (groupings, left-to-right or top-to-bottom flow, swim-lanes, etc.).
- Flag any ambiguities or gaps that the Planner will need to resolve.

### Explorer deliverable
A concise written summary passed back to the main agent containing:
- List of figure components and their relationships
- Extracted style rules (colors, icon vocabulary, font, edge style) from the reference figure
- Layout constraints derived from the crude sketch
- Open questions or missing information for the Planner

The main agent passes this summary in full to the Planner. **The Planner must not begin planning until it has received this summary.**

---

## Phase 2 — Plan

Spawn the **Planner** subagent with the Explorer's summary and all original user-provided resources.

### Planner objectives
- Design the complete figure: spatial layout, component placement, labels, arrows, groupings, and color assignments.
- Specify which stand-alone icons or visual elements must be generated externally (e.g., a rendered mesh thumbnail, an annotated parameter table icon). For each such element, provide a clear generation prompt.
- Determine which elements can be drawn natively in draw.io vs. which require raster images embedded as `image;` cells.
- Produce a directive that is detailed enough for both the Element generator and the Drawer to act on independently.

### Planner deliverable
A structured figure directive written to `<output_directory>/figure_plan.md` containing:
- Figure dimensions and orientation (landscape/portrait)
- Component list with approximate positions (row/column or bounding box estimates)
- Label text for each component and edge
- Color assignments (hex codes or named tokens from the reference style)
- Element generation prompts: one per icon/image to be produced by GPT Image 2, including desired size and background
- Draw.io construction notes: shape types, swimlane groupings, edge routing hints

---

## Phase 3 — Element generation

Spawn the **Element generator** subagent with the Planner's directive (`figure_plan.md`) and all original user-provided resources.

### Element generator objectives
- For each icon or visual element specified in the Planner's directive, generate the image using GPT Image 2.
- Save every generated file to `<output_directory>/elements/`, using descriptive filenames (e.g., `medical_bag_mesh_icon.png`).
- Match the visual style (palette, rendering quality, background transparency if needed) to the reference architecture figure.

### Element generator deliverable
A manifest of generated files (name → file path → brief description) passed back to the main agent, which then forwards it to the Drawer alongside the Planner's directive.

---

## Phase 4 — Draw (initial)

Spawn the **Drawer** subagent with:
- The Planner's directive (`figure_plan.md`)
- The element manifest from the Element generator
- All original user-provided resources
- The `drawio-skill` instructions (the Drawer must follow the `drawio-skill` workflow for XML generation, CLI export, and self-check)

### Drawer objectives
- Construct the draw.io XML according to the Planner's layout directive.
- Embed generated icons as `image;` cells referencing their saved paths.
- Export a draft PNG (without `-e`) to `<output_directory>/` using the draw.io CLI.
- Run the `drawio-skill` self-check on the draft PNG; auto-fix any obvious visual issues (overlapping shapes, clipped labels, missing connections).
- Save the `.drawio` source file to `<output_directory>/`.

### Drawer deliverable
- `<output_directory>/<figure_name>.drawio` — editable source
- `<output_directory>/<figure_name>.png` — draft PNG for review

The main agent passes the draft PNG path and the `.drawio` path to the Planner for review.

---

## Phase 5 — Review loop

The main agent enters a **draw-review-draw** cycle driven entirely by the Planner.

### Each review round

1. **Main agent** passes the current draft PNG to the **Planner**.
2. **Planner** reviews the figure against the original crude layout, the reference style, and the figure directive. It checks:
   - All specified components are present and correctly labeled.
   - Layout matches the crude sketch's spatial intent.
   - Colors and icon styles match the reference architecture figure.
   - Arrows and data flows are correctly represented.
   - No components are missing, mislabeled, or visually cluttered.
3. If the Planner is **satisfied**, the review loop ends — proceed to Phase 6.
4. If the Planner identifies issues, it produces a **fix directive**: a ranked list of specific, targeted changes (e.g., "move the 'Physics Solver' box to the right of the 'Mesh Builder' box", "replace the orange fill on the output node with hex #4A90D9", "add a dashed arrow from 'Segmentation' to 'Param Inference'").
5. **Main agent** forwards the fix directive to the **Element generator** (if new or revised icons are needed) and to the **Drawer** (for XML edits and re-export).
6. **Drawer** applies the targeted XML edits (following `drawio-skill` edit rules: minimal changes for single-element fixes, full regeneration only for layout-wide changes), re-exports the PNG, and returns the updated PNG path to the main agent.
7. Return to step 1.

### Loop safety valve
- After **5 review rounds** with unresolved issues, the Planner must summarize what remains unsatisfactory and the main agent must surface this to the human user, offering to open the `.drawio` file in draw.io desktop for manual fine-tuning.

---

## Phase 6 — Final export

Once the Planner approves the figure:

1. The **Drawer** re-exports the approved diagram using `-e` (embedded XML) to produce `<output_directory>/<figure_name>.drawio.png`.
2. Immediately after the `-e` PNG export, the Drawer runs `python3 <drawio-skill-dir>/scripts/repair_png.py <output_directory>/<figure_name>.drawio.png` to fix the truncated IEND chunk (draw.io CLI issue #8).
3. The main agent reports all output file paths to the user:
   - `<output_directory>/<figure_name>.drawio` — editable source
   - `<output_directory>/<figure_name>.drawio.png` — final PNG with embedded diagram
   - `<output_directory>/elements/` — all generated icon files
   - `<output_directory>/figure_plan.md` — Planner's directive (for future reference)
4. Offer to open the `.drawio` file in draw.io desktop for fine-tuning: `xdg-open` on Linux, `open` on macOS, `start` on Windows.
