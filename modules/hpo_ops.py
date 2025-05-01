import re
import subprocess
import time
from pathlib import Path
import os
import warnings
import logging
import shutil
from typing import Optional

from .utils import log, DeepSeekClient
from .text_ops import write_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=r"The current process just got forked")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

DOCKER_TIMEOUT = 60 * 60


def _build_tag_script(sample: str, script_path: Path) -> None:
    script_path.write_text(
        f"""#!/usr/bin/bash
cd /PhenoTagger/src/
rm -f /PhenoTagger/example/input/*
cp /mnt/{sample}.PubTator /PhenoTagger/example/input/
python /PhenoTagger/src/PhenoTagger_tagging.py -i ../example/input/ -o ../output/ || exit 1
shopt -s nullglob
cp ../output/* /mnt/ 2>/dev/null || true
""",
        encoding="utf-8",
    )

def _check_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker binary not found in PATH")
    subprocess.check_output(["docker", "info"], stderr=subprocess.STDOUT, timeout=5)


def execute_phenotagger(text: str, sample_name: str, result_dir: Path) -> str:
    _check_docker()

    input_pubtator = result_dir / f"{sample_name}.PubTator"
    input_pubtator.write_text(f"1|t|description\n1|a|{text}\n\n\n", encoding="utf-8")

    script_path = result_dir / f"{sample_name}.sh"
    _build_tag_script(sample_name, script_path)
    subprocess.run(["chmod", "+x", script_path.as_posix()], check=True)

    cmd = [
        "docker",
        "run",
        "--user",
        "root",
        "--rm",
        "-v",
        f"{result_dir.resolve()}:/mnt",
        "albertea/phenotagger:1.2",
        f"/mnt/{script_path.name}",
        "--gpus",
        "all",
    ]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT)
    runtime = time.time() - start
    log.info(f"PhenoTagger finished in {runtime:.1f}s (rc={proc.returncode})")

    if proc.returncode != 0:
        raise RuntimeError(f"PhenoTagger failed: {proc.stderr.strip()}")

    parse_cmd = (
        f'grep "^1" {input_pubtator.name} | sed "1,2d" | cut -f4,5 | '
        'sed "s+^+*+" | sed "s+\\tHP:+*\\tHP:+"'
    )
    try:
        hpo = subprocess.check_output(parse_cmd, shell=True, cwd=result_dir).decode().strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to parse PhenoTagger output")

    if not hpo:
        raise RuntimeError("PhenoTagger returned no HPO terms")

    (result_dir / f"{sample_name}_03_raw_hpo.txt").write_text(hpo, encoding="utf-8")

    numbered_pubtator = result_dir / f"{sample_name}_03_phenotagger.PubTator"
    if numbered_pubtator.exists():
        numbered_pubtator.unlink()
    input_pubtator.rename(numbered_pubtator)

    neg2_src = result_dir / f"{sample_name}.neg2.PubTator"
    if neg2_src.exists():
        neg2_dst = result_dir / f"{sample_name}_03_phenotagger.neg2.PubTator"
        if neg2_dst.exists():
            neg2_dst.unlink()
        neg2_src.rename(neg2_dst)

    return hpo



def get_hpo(text: str, chat: DeepSeekClient, sample_name: str, result_dir: Path) -> str:
    text = text.replace("\n", " ")
    hpo = execute_phenotagger(text, sample_name, result_dir)
    write_text(hpo, "_03_hpo_terms", sample_name, result_dir)
    return hpo


def filter_terms(
    hpo_terms: str,
    text: str,
    chat: DeepSeekClient,
    sample_name: str,
    result_dir: Path,
) -> Optional[str]:
    if not hpo_terms:
        raise ValueError("Empty HPO term list")

    prompt = (
        "Analyze the patient text and match it with the provided HPO term list.\n"
        "Rules:\n"
        "1. Use only terms present in the list; do not invent new ones.\n"
        "2. Preserve HPO codes exactly.\n"
        "3. Remove terms that do not describe the patient.\n"
        '4. Output each kept term on its own line as "Term name HP:XXXXXXX".'
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
