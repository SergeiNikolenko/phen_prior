from pathlib import Path
import asyncio
import shutil
from typing import Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor

from joblib import cpu_count
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
    Group,
)
from rich.live import Live

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
        api_key: Optional[str],
        output_dir: Path,
        show_progress: bool = True,
        tail_cb: Optional[Callable[[str], None]] = None,
    ):
        self.med_doc = med_doc
        self.sqlite_path = sqlite_path
        self.sample_name = sqlite_path.stem.split(".", 1)[0]
        self.result_dir = output_dir
        self.show_progress = show_progress
        self.chat = DeepSeekClient(api_key=api_key)
        if tail_cb:
            self.chat.tail_cb = tail_cb

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
            tail_task = ctx.add_task("", total=None)

            def _tail_update(t: str):
                ctx.update(
                    tail_task,
                    description=t.replace("\n", " ").replace("\r", " ")[:100],
                )

            self.chat.tail_cb = _tail_update

        def step(msg: str):
            log.info(msg)
            if ctx:
                ctx.update(steps, advance=1, description=msg)

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


def _is_output_complete(out_dir: Path, sample_name: str) -> bool:
    expected = [
        out_dir / f"{sample_name}_02_processed_text.txt",
        out_dir / f"{sample_name}_04_filtered_terms.txt",
        out_dir / f"{sample_name}_05_clinprior.csv",
    ]
    return all(p.exists() for p in expected)


def _run_pipeline_sync(
    med_doc: Path,
    sqlite_path: Path,
    output_root: Path,
    api_key: Optional[str],
    bar_progress: Progress,
    bar_id: int,
    tail_progress: Progress,
    tail_id: int,
):
    sample_name = sqlite_path.stem.split(".", 1)[0]
    out_dir = output_root / f"result_{sample_name}" / med_doc.stem
    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _tail_update(msg: str):
            tail_progress.update(tail_id, description=msg[:100])

        Pipeline(
            med_doc,
            sqlite_path,
            api_key,
            out_dir,
            show_progress=False,
            tail_cb=_tail_update,
        ).run()
        bar_progress.update(bar_id, advance=1)
        return True
    except Exception as e:
        (out_dir / "error.txt").write_text(str(e))
        log.error(f"FAILED {med_doc}: {e}")
        bar_progress.update(bar_id, advance=1)
        return False


async def _process_doc_async(
    sem: asyncio.Semaphore,
    loop,
    executor,
    med_doc: Path,
    sqlite_path: Path,
    output_root: Path,
    api_key: Optional[str],
    bar_progress: Progress,
    bar_id: int,
    tail_progress: Progress,
    tail_id: int,
):
    async with sem:
        return await loop.run_in_executor(
            executor,
            _run_pipeline_sync,
            med_doc,
            sqlite_path,
            output_root,
            api_key,
            bar_progress,
            bar_id,
            tail_progress,
            tail_id,
        )


async def _batch_async(
    docs_dir: Path,
    sqlite_path: Path,
    output_root: Optional[Path],
    config: Optional[Path],
    log_level: str,
    workers: int,
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
    setup_logging_file_only(output_root / "phen_prior.log", log_level)
    api_key = load_config(config)

    sample_name = sqlite_path.stem.split(".", 1)[0]
    pending_docs: List[Path] = []
    completed = 0
    for doc in docs:
        out_dir = output_root / f"result_{sample_name}" / doc.stem
        if out_dir.exists() and _is_output_complete(out_dir, sample_name):
            completed += 1
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        pending_docs.append(doc)

    total = len(docs)
    remaining = len(pending_docs)

    console = Console()
    console.print(f"Total docs: {total} | Completed: {completed} | Remaining: {remaining}")
    if not pending_docs:
        console.print("Nothing to process. Exiting.")
        raise typer.Exit()

    sem = asyncio.Semaphore(workers)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=min(workers, cpu_count()))

    bar_progress = Progress(
        SpinnerColumn(),
        BarColumn(bar_width=None),
        TaskProgressColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    tail_progress = Progress(TextColumn("{task.description}"), console=console)

    bar_id = bar_progress.add_task("Processing", total=remaining)
    tail_id = tail_progress.add_task("", total=None)

    with Live(Group(bar_progress, tail_progress), console=console, refresh_per_second=10):
        coros = [
            _process_doc_async(
                sem,
                loop,
                executor,
                doc,
                sqlite_path,
                output_root,
                api_key,
                bar_progress,
                bar_id,
                tail_progress,
                tail_id,
            )
            for doc in pending_docs
        ]
        results = await asyncio.gather(*coros)

    failed = results.count(False)
    console.print(f"Finished. OK: {remaining - failed} | Failed: {failed}")


@app.command()
def batch(
    docs_dir: Path = typer.Option(BASE_DIR / "../med_docs", "-d", "--docs-dir"),
    sqlite_path: Path = typer.Option(BASE_DIR / "../FND00006610.vcf.sqlite", "-s", "--sqlite"),
    output_root: Optional[Path] = typer.Option(None, "-o", "--output-root"),
    config: Optional[Path] = typer.Option(BASE_DIR / "data/tokenizer_config.json", "-c", "--config"),
    log_level: str = typer.Option("info", "--log-level"),
    workers: int = typer.Option(4, "-w", "--workers"),
):
    asyncio.run(
        _batch_async(
            docs_dir,
            sqlite_path,
            output_root,
            config,
            log_level,
            workers,
        )
    )


@app.command()
def run(
    med_doc: Path = typer.Option(BASE_DIR / "../med_docs_test/test.txt", "-m", "--med_doc"),
    sqlite_path: Path = typer.Option(BASE_DIR / "../FND00006610.vcf.sqlite", "-s", "--sqlite"),
    config: Optional[Path] = typer.Option(BASE_DIR / "data/tokenizer_config.json", "-c", "--config"),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output_dir"),
    log_level: str = typer.Option("info", "--log-level"),
    override: bool = typer.Option(False, "--override"),
):
    for p in (med_doc, sqlite_path):
        check_file_exists(p)

    sample_name = sqlite_path.stem.split(".", 1)[0]
    if output_dir is None:
        output_dir = BASE_DIR / sample_name

    if output_dir.exists():
        if override:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        else:
            log.error(f"Output directory {output_dir} already exists. Use --override to overwrite.")
            raise typer.Exit(code=1)
    else:
        output_dir.mkdir(parents=True)

    setup_logging_file_only(output_dir / "phen_prior.log", log_level)
    api_key = load_config(config)

    Pipeline(
        med_doc,
        sqlite_path,
        api_key,
        output_dir,
        show_progress=True,
    ).run()


if __name__ == "__main__":
    app()
