---
name: add-papers-to-warehouse
description: Prompt the user for a list of paper titles, download their PDFs, compress them, and move the compressed PDFs to docs/paper_warehouse/.
---

# add-papers-to-warehouse

## Goal

For each paper the user provides, download its PDF, compress it, and store it under `docs/paper_warehouse/` with a standardised filename.

---

## Step 1 — Collect paper titles

Either (1) ask the human user to enter the list of paper titles they want to add to the warehouse. Wait for their response before proceeding or (2) if the user has already provided a source that includes a list of paper titles when invoking the skill, use that list directly.

---

## Step 2 — For each paper, determine metadata

Before downloading, determine the following three pieces of metadata for each paper (use a web search if needed):

| Field | Description | Example |
|-------|-------------|---------|
| `FAMILY` | Family (last) name of the **first** author | `Chen` |
| `YEAR` | Four-digit publication year | `2024` |
| `SHORTHAND` | A concise snake\_case identifier for the paper (e.g., abbreviated title or common nickname) | `droid_slam` |

(Optional) Ask the user to confirm or correct the metadata for each paper before downloading. Do this only if the user has explicitly required metadata confirmation when invoking the skill. Otherwise, proceed with the best-effort metadata you have determined.

---

## Step 3 — Ensure Ghostscript is installed

Check whether `gs` (Ghostscript) is available:

```bash
gs --version
```

If the command is not found, notify the user that Ghostscript is required for PDF compression and ask them to install it before continuing:

```
Ghostscript is not installed. Please install it and re-run the skill.
  Ubuntu/Debian: sudo apt install ghostscript
  Arch:          sudo pacman -S ghostscript
  macOS:         brew install ghostscript
```

Do not proceed until `gs` is available.

---

## Step 4 — Download PDF

For each paper:

1. Locate a publicly available PDF (try arXiv, Semantic Scholar, the venue page, or the authors' project page, in that order).
2. Download the PDF to `.agents/tmp/` with an arbitrary temporary name (e.g., the paper's arXiv ID or a sanitised title):

```bash
mkdir -p .agents/tmp
curl -L "<pdf_url>" -o ".agents/tmp/<temp_name>.pdf"
```

Verify that the downloaded file is a valid PDF (non-zero size, starts with `%PDF`).

---

## Step 5 — Compress the PDF

The compress script (`docs/scripts/compress_pdf_linux.sh`) expects **both** the input and output files to live under `docs/paper_warehouse/`. Follow these sub-steps:

1. Move the downloaded PDF into `docs/paper_warehouse/` under the temporary name:

```bash
mv ".agents/tmp/<temp_name>.pdf" "docs/paper_warehouse/<temp_name>.pdf"
```

2. Run the compress script from the **repository root**, passing the input stem and the desired output stem (no `.pdf` extension):

```bash
bash docs/scripts/compress_pdf_linux.sh \
    "<temp_name>" \
    "<FAMILY>_<YEAR>_<SHORTHAND>_compressed"
```

This produces `docs/paper_warehouse/<FAMILY>_<YEAR>_<SHORTHAND>_compressed.pdf`.

3. Remove the uncompressed original from `docs/paper_warehouse/`:

```bash
rm "docs/paper_warehouse/<temp_name>.pdf"
```

---

## Step 6 — Clean up `tmp/`

After all papers have been processed, ensure `tmp/` contains no leftover PDF files from this run:

```bash
# Should already be empty for each paper, but verify:
ls .agents/tmp/*.pdf 2>/dev/null && rm .agents/tmp/*.pdf || true
```

---

## Step 7 — Report results

Report to the user:
- Which papers were successfully added (final filename in `docs/paper_warehouse/`).
- Which papers could not be found or downloaded, and why.
- Any papers whose metadata was uncertain and was assumed.

---

## Naming convention reference

```
<FAMILY>_<YEAR>_<SHORTHAND>_compressed.pdf
```

Examples:
- `Teed_2021_droid_slam_compressed.pdf`
- `Kerbl_2023_3dgs_compressed.pdf`
- `Ho_2020_ddpm_compressed.pdf`
