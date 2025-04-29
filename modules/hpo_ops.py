# modules/hpo_ops.py
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
import os
import warnings
import logging

import shutil




from .utils import log, DeepSeekClient
from .text_ops import write_text, translate_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=r"The current process just got forked")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

SAMPLE_NAME = None
RESULT_DIR = None


def _build_tag_script(sample: str, script_path: Path):
    script_path.write_text(
        f"""#!/usr/bin/bash
cd /PhenoTagger/src/
rm /PhenoTagger/example/input/*
cp /mnt/{sample}.PubTator /PhenoTagger/example/input/
python /PhenoTagger/src/PhenoTagger_tagging.py -i ../example/input/ -o ../output/
cp ../output/* /mnt/
""",
        encoding="utf-8",
    )




DOCKER_TIMEOUT = 60 * 60          # 1 час

def _check_docker():
    if shutil.which("docker") is None:
        raise RuntimeError("Docker binary not found in PATH")
    try:
        subprocess.check_output(["docker", "info"], stderr=subprocess.STDOUT, timeout=5)
    except Exception as e:
        raise RuntimeError("Docker daemon is not running") from e

def execute_phenotagger(text: str) -> str:
    _check_docker()

    pubtator_path = RESULT_DIR / f"{SAMPLE_NAME}.PubTator"
    pubtator_path.write_text(f"1|t|description\n1|a|{text}\n\n\n", encoding="utf-8")

    script_path = RESULT_DIR / f"{SAMPLE_NAME}.sh"
    _build_tag_script(SAMPLE_NAME, script_path)
    subprocess.run(["chmod", "+x", script_path.as_posix()], check=True)

    log.info("PhenoTagger: docker run started")
    cmd = [
        "docker", "run", "--user", "root", "--rm",
        "-v", f"{RESULT_DIR.resolve()}:/mnt",
        "albertea/phenotagger:1.2",
        f"/mnt/{script_path.name}",
        "--gpus", "all"
    ]

    start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_line = time.time()
    try:
        while True:
            line = p.stdout.readline()
            if line:
                log.debug(f"PhenoTagger | {line.rstrip()}")
                last_line = time.time()
            if p.poll() is not None:
                break
            if time.time() - last_line > 30:
                log.info(f"PhenoTagger: still running ({int(time.time()-start)} s)")
                last_line = time.time()
            if time.time() - start > DOCKER_TIMEOUT:
                p.kill()
                raise RuntimeError("PhenoTagger timeout exceeded")
        rc = p.wait()
    finally:
        if p.stdout:
            p.stdout.close()

    if rc != 0:
        raise RuntimeError("PhenoTagger exited with non-zero status")

    log.info(f"PhenoTagger finished in {time.time()-start:.1f} s, parsing output")
    cmd_parse = (
        f'grep ^1 {pubtator_path.name} | sed 1,2d | cut -f 4,6 | '
        'sed "s+^+*+" | sed "s+\\tHP:+*\\tHP:+"'
    )
    try:
        return subprocess.check_output(cmd_parse, shell=True, cwd=RESULT_DIR).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        log.error("Failed to parse PhenoTagger output")
        raise


def get_hpo(text: str, chat: DeepSeekClient, sample_name: str, result_dir: Path) -> str:
    global SAMPLE_NAME, RESULT_DIR
    SAMPLE_NAME = sample_name
    RESULT_DIR = result_dir

    log.info("Translation to English started")
    t0 = time.time()
    eng = translate_text(text, chat)
    log.info(f"Translation finished in {time.time() - t0:.1f}s")
    write_text(eng, "_eng", sample_name, result_dir)

    hpo = execute_phenotagger(eng)
    write_text(hpo, "_hpo_terms", sample_name, result_dir)
    return hpo


def filter_terms(hpo_terms: str, text: str, chat: DeepSeekClient, sample_name: str, result_dir: Path) -> str | None:
    prompt = (
        "Проанализируй текст и сопоставь его с предоставленным набором HPO-термов.\n"
        "Строго соблюдай правила:\n"
        "1. Используй только термы из списка; не добавляй новые.\n"
        "2. Коды HPO не изменяй.\n"
        "3. Удали термы, не относящиеся к пациенту.\n"
        "4. Формат: «Название терма код» на каждой новой строке."
    )
    filtered = chat.ask(f"{text}\n{hpo_terms}", prompt, temperature=0.0)
    write_text(filtered, "_filtered_terms", sample_name, result_dir)
    codes = re.findall(r"HP:\d{7}", filtered)
    return ",".join(codes) if codes else None


def execute_clinprior(terms: str, sample_name: str, result_dir: Path):
    r_script = result_dir / "clinprior_script.r"
    if not r_script.exists():
        raise FileNotFoundError(f"R-script not found: {r_script}")

    log.info("ClinPrior: docker run started")
    t0 = time.time()

    cmd = [
        "docker", "run",
        "--platform", "linux/amd64",
        "--rm",
        "-v", f"{result_dir.resolve()}:/mnt",
        "aschluterclinprior/clinprior2:latest",
        "bash", "-c",
        f"Rscript /mnt/{r_script.name} {terms} {sample_name}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.time() - t0
    log.info(f"ClinPrior: docker finished in {runtime:.1f}s (rc={result.returncode})")

    log.debug("ClinPrior stdout:\n" + result.stdout.strip())
    if result.stderr:
        log.debug("ClinPrior stderr:\n" + result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError("ClinPrior exited with non-zero status")

    csv_path = result_dir / f"{sample_name}_clinprior.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ClinPrior CSV not found: {csv_path}")