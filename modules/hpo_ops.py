from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .utils import log, DeepSeekClient
from .text_ops import write_text


def get_hpo(text: str, chat: DeepSeekClient, sample_name: str, result_dir: Path) -> str:
    text = text.replace("\n", " ")
    prompt = (
        "You are an experienced clinical geneticist. "
        "Read the patient text and return a unique list of phenotype terms from the Human Phenotype Ontology. "
        'Each line must follow the format "Full term name HP:0000000". '
        "Provide no additional commentary or metadata."
    )
    response = chat.ask(text, prompt, temperature=0.1)
    write_text(response, "_03_hpo_terms", sample_name, result_dir)
    return response


def filter_terms(
    hpo_terms: str,
    text: str,
    chat: DeepSeekClient,
    sample_name: str,
    result_dir: Path,
) -> str | None:
    if not hpo_terms:
        raise ValueError("Empty HPO term list")
    prompt = (
        "Match the patient text to the provided HPO term list.\n"
        "Rules:\n"
        "1. Use only terms from the list; do not add new ones.\n"
        "2. Preserve HPO codes exactly.\n"
        "3. Remove terms that do not describe the patient.\n"
        "4. \n"
        '5. Output each remaining term on its own line in the form "Term name HP:XXXXXXX".'
    )
    response = chat.ask(f"{text}\n\nHPO term list:\n{hpo_terms}", prompt, temperature=0.0)
    write_text(response, "_04_filtered_terms", sample_name, result_dir)
    codes = re.findall(r"HP:\d{7}", response)
    return ",".join(dict.fromkeys(codes)) if codes else None


def execute_clinprior(terms: str, sample_name: str, result_dir: Path) -> None:
    if not terms:
        raise ValueError("No HPO terms for ClinPrior")
    r_script = result_dir / "clinprior_script.r"
    if not r_script.exists():
        raise FileNotFoundError(f"R-script not found: {r_script}")
    log.info("ClinPrior: docker run started")
    t0 = time.time()
    cmd = [
        "docker",
        "run",
        "--platform",
        "linux/amd64",
        "--rm",
        "-v",
        f"{result_dir.resolve()}:/mnt",
        "aschluterclinprior/clinprior2:latest",
        "bash",
        "-c",
        f"Rscript /mnt/{r_script.name} {terms} {sample_name}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.time() - t0
    log.info(f"ClinPrior finished in {runtime:.1f}s (rc={proc.returncode})")
    if proc.returncode != 0:
        raise RuntimeError(f"ClinPrior failed: {proc.stderr.strip()}")
    csv_path = result_dir / f"{sample_name}_clinprior.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ClinPrior CSV not found: {csv_path}")
    csv_path.rename(result_dir / f"{sample_name}_05_clinprior.csv")
