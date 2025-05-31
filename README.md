# Phenotype Prioritizer

Automated pipeline that cleans clinical text, extracts HPO terms, runs ClinPrior, and re-orders an OpenCRAVAT SQLite variant file so the most phenotype-relevant variants appear first.  
Includes a helper script to anonymize raw notes before analysis.

---

## Prerequisites

* Python 3.11+
* Docker (for PhenoTagger v1.2 and ClinPrior)
* OpenAI API key
* UV package manager

```bash
pip install uv        # one-time install
```

---

## Installation

```bash
uv sync          # create env & install all deps
# optional: uv add <pkg>  # install extra libs
```

Place your OpenAI key either in `OPENAI_API_KEY` env-var
or in `data/tokenizer_config.json` as
`{ "openai_api_key": "sk-…" }`.

---

## Quick Start

### Single document

```bash
uv run main.py run \
  --med_doc  note.txt \
  --sqlite   sample.vcf.sqlite \
  --output_dir results
```

### Batch folder

```bash
uv run main.py batch \
  --docs-dir  med_docs/ \
  --sqlite    sample.vcf.sqlite \
  --output-root batch_results
```

`run` processes one note; `batch` processes every `.txt` in a folder (parallel workers default = 4).

---

## Outputs

* `<sample>_02_processed_text.txt` – cleaned & translated note
* `<sample>_03_hpo_terms.txt`     – raw HPO list
* `<sample>_04_filtered_terms.txt` – final HPO list
* `<sample>_05_clinprior.csv`     – gene rankings
* updated `sample.vcf.sqlite` with variants re-ordered by ACMG + phenotype relevance
* `phen_prior.log` for full trace

---

## Anonymize Utility

Combine and de-identify notes before running the pipeline.

```bash
uv run anonymize.py \
  --input  med_docs/ \
  --output anonymized/
```

Creates `anonymized/<folder>_combined.txt` files with names, dates and other PII masked.

---

## Pipeline Summary

1. **Text Processing** – translate RU→EN, expand abbreviations, remove noise.
2. **HPO Extraction** – PhenoTagger in Docker.
3. **HPO Filtering** – GPT-4 removes irrelevant terms.
4. **Gene Prioritization** – ClinPrior in Docker.
5. **Variant Re-ordering** – SQLite updated by ACMG class + gene rank.

---

## Project Structure

```
.
├── main.py          # CLI: run / batch
├── anonymize.py     # PII removal helper
├── modules/
│   ├── text_ops.py     # GPT-based cleaning
│   ├── hpo_ops.py      # PhenoTagger & ClinPrior
│   ├── db_ops.py       # SQLite re-ordering
│   └── utils.py        # config, logging
└── pyproject.toml   # dependencies
```

Ready to prioritize variants based on patient phenotype in one command.


