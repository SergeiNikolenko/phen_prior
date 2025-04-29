from pathlib import Path
import re
import shutil
import multiprocessing
from typing import Optional, List
from joblib import Parallel, delayed
import nltk
import typer

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from modules.utils import (
    load_config,
    DeepSeekClient,
    log,
    check_file_exists,
    setup_logging_file_only,
)
from modules.text_ops import process_text
from modules.hpo_ops import get_hpo, filter_terms, execute_clinprior
from modules.db_ops import modify_sqlite

app = typer.Typer(add_help_option=False)
BASE_DIR = Path(__file__).parent


class Pipeline:
    def __init__(
        self,
        med_doc: Path,
        sqlite_path: Path,
        config_path: Optional[Path],
        output_dir: Path,
        log_level: str,
        show_progress: bool = True,
    ):
        self.med_doc = med_doc
        self.sqlite_path = sqlite_path
        self.sample_name = sqlite_path.stem
        api_key = load_config(config_path)
        self.chat = DeepSeekClient(api_key=api_key)
        self.result_dir = output_dir
        self.show_progress = show_progress
        self.log = setup_logging_file_only(self.result_dir / "pipeline.log", log_level)

    def run(self):
        nltk.download("punkt", quiet=True)
        ctx = (
            Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            if self.show_progress
            else None
        )

        if ctx:
            ctx.__enter__()
            steps = ctx.add_task("Step 1/6: Reading medical doc", total=6)

        def step(msg):
            log.info(msg)
            if ctx:
                ctx.update(steps, advance=1, description=msg)

        step("Step 1/6: Reading medical doc")
        text = self.med_doc.read_text(encoding="utf-8")

        step("Step 2/6: Processing text")
        processed = process_text(text, self.chat, self.sample_name, self.result_dir)

        step("Step 3/6: Extracting HPO")
        hpo_terms = get_hpo(processed, self.chat, self.sample_name, self.result_dir)

        step("Step 4/6: Filtering terms")
        filtered = filter_terms(
            hpo_terms, processed, self.chat, self.sample_name, self.result_dir
        )

        step("Step 5/6: Running ClinPrior")
        self._execute_clinprior(filtered)

        step("Step 6/6: Modifying SQLite")
        modify_sqlite(self.sqlite_path, self.sample_name, self.result_dir)

        if ctx:
            ctx.update(steps, advance=1, description="Done")
            ctx.__exit__(None, None, None)
        log.info("Pipeline completed.")

    def _execute_clinprior(self, filtered_terms: Optional[str]):
        wl = self._load_whitelist()
        final_terms = (
            ",".join(t for t in filtered_terms.split(",") if t in wl)
            if filtered_terms
            else "HP:0000118"
        )
        src = BASE_DIR / "clinprior_script.r"
        dst = self.result_dir / "clinprior_script.r"
        if not dst.exists():
            shutil.copy(src, dst)
        execute_clinprior(final_terms, self.sample_name, self.result_dir)

    def _load_whitelist(self):
        wl_path = Path("data") / "hpo_whitelist.txt"
        check_file_exists(wl_path)
        return {l.strip() for l in wl_path.read_text(encoding="utf-8").splitlines() if l.strip()}


def _collect_docs(folder: Path) -> List[Path]:
    return list(folder.rglob("*.txt"))


def _process_doc(
    med_doc: Path,
    sqlite_path: Path,
    output_root: Path,
    config: Optional[Path],
    log_level: str,
    progress: Progress,
    task_id: int,
):
    sample_name = sqlite_path.stem
    out_dir = output_root / f"result_{sample_name}" / med_doc.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    Pipeline(
        med_doc,
        sqlite_path,
        config,
        out_dir,
        log_level,
        show_progress=False,
    ).run()
    progress.update(task_id, advance=1)


@app.command()
def batch(
    docs_dir: Path = typer.Option(
        BASE_DIR / "../med_docs",
        "-d",
        "--docs-dir",
        help="Directory with .txt files (recursive)",
    ),
    sqlite_path: Path = typer.Option(
        BASE_DIR / "../FND00006610.vcf.sqlite",
        "-s",
        "--sqlite",
        help="Path to single sqlite used for all documents",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output-root",
        help="Root directory for result_* folders (default: one level above docs_dir)",
    ),
    config: Optional[Path] = typer.Option(
        BASE_DIR / "data/tokenizer_config.json",
        "-c",
        "--config",
        help="Path to JSON with API key",
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="Log level (debug, info, warning, error)"
    ),
    workers: int = typer.Option(
        4,
        "-w",
        "--workers",
        help="Number of parallel workers (default = 4)",
    ),
):
    for p in (docs_dir, sqlite_path):
        check_file_exists(p)

    docs = _collect_docs(docs_dir)
    if not docs:
        log.error("No .txt documents found")
        raise typer.Exit(code=1)

    if output_root is None:
        output_root = docs_dir.parent
    output_root.mkdir(parents=True, exist_ok=True)

    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("Processing files:"),
        BarColumn(bar_width=None),
        TaskProgressColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files:", total=len(docs))
        Parallel(n_jobs=workers, prefer="threads")(
            delayed(_process_doc)(
                doc,
                sqlite_path,
                output_root,
                config,
                log_level,
                progress,
                task,
            )
            for doc in docs
        )


@app.command()
def run(
    med_doc: Path = typer.Option(
        BASE_DIR / "../med_docs_test/test.txt",
        "-m",
        "--med_doc",
        help="Path to medical document",
    ),
    sqlite_path: Path = typer.Option(
        BASE_DIR / "../FND00006610.vcf.sqlite",
        "-s",
        "--sqlite",
        help="Path to sqlite",
    ),
    config: Optional[Path] = typer.Option(
        BASE_DIR / "data/tokenizer_config.json",
        "-c",
        "--config",
        help="Path to JSON with API key",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output_dir",
        help="Directory for result files (default: <sample_name>)",
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="Log level (debug, info, warning, error)"
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Overwrite output directory if it already exists",
    ),
):
    for p in (med_doc, sqlite_path):
        check_file_exists(p)

    sample_name = sqlite_path.stem
    if output_dir is None:
        output_dir = BASE_DIR / sample_name

    if output_dir.exists():
        if override:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        else:
            log.error(
                f"Output directory {output_dir} already exists. Use --override to overwrite."
            )
            raise typer.Exit(code=1)
    else:
        output_dir.mkdir(parents=True)

    Pipeline(
        med_doc,
        sqlite_path,
        config,
        output_dir,
        log_level,
        show_progress=True,
    ).run()


if __name__ == "__main__":
    app()