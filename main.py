# main.py
from pathlib import Path
import re
import shutil
import nltk
import typer
from typing import Optional

from modules.utils import (
    load_config,
    DeepSeekClient,
    log,
    check_file_exists,
    setup_logging_file_only
)
from modules.text_ops import process_text
from modules.hpo_ops import get_hpo, filter_terms, execute_clinprior
from modules.db_ops import modify_sqlite
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
    ):
        self.med_doc = med_doc
        self.sqlite_path = sqlite_path
        self.sample_name = re.sub(r"\.(vcf|gz|sqlite)$", "", sqlite_path.name)
        api_key = load_config(config_path)
        self.chat = DeepSeekClient(api_key=api_key)
        self.result_dir = output_dir
        setup_logging_file_only(self.result_dir / "pipeline.log", log_level)

    def run(self):
        nltk.download("punkt", quiet=True)
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            steps = progress.add_task("Step 1/6: Reading medical doc", total=6)
            log.info("Step 1/6: Reading medical doc")
            text = self.med_doc.read_text(encoding="utf-8")
            
            progress.update(steps, advance=1, description="Step 2/6: Processing text")
            log.info("Step 2/6: Processing text")
            processed = process_text(text, self.chat, self.sample_name, self.result_dir)

            progress.update(steps, advance=1, description="Step 3/6: Extracting HPO")
            log.info("Step 3/6: Extracting HPO")
            hpo_terms = get_hpo(processed, self.chat, self.sample_name, self.result_dir)

            progress.update(steps, advance=1, description="Step 4/6: Filtering terms")
            log.info("Step 4/6: Filtering terms")
            filtered = filter_terms(hpo_terms, processed, self.chat, self.sample_name, self.result_dir)
            
            progress.update(steps, advance=1, description="Step 5/6: Running ClinPrior")
            log.info("Step 5/6: Running ClinPrior")
            self._execute_clinprior(filtered)

            progress.update(steps, advance=1, description="Step 6/6: Modifying SQLite")
            log.info("Step 6/6: Modifying SQLite")
            modify_sqlite(self.sqlite_path, self.sample_name, self.result_dir)

            progress.update(steps, advance=1, description="Done")
        log.info("Pipeline completed.")


    def _execute_clinprior(self, filtered_terms: Optional[str]):
        wl = self._load_whitelist()
        final_terms = ",".join(t for t in filtered_terms.split(",") if t in wl) if filtered_terms else "HP:0000118"


        src = BASE_DIR / "clinprior_script.r"
        dst = self.result_dir / "clinprior_script.r"
        
        if not dst.exists():
            shutil.copy(src, dst)
        
        

        execute_clinprior(final_terms, self.sample_name, self.result_dir)


    def _load_whitelist(self):
        wl_path = Path("data") / "hpo_whitelist.txt"
        check_file_exists(wl_path)
        return {l.strip() for l in wl_path.read_text(encoding="utf-8").splitlines() if l.strip()}


@app.command()
def run(
    med_doc: Path = typer.Option(
        BASE_DIR / "../med_docs/test.txt",
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

    sample_name = re.sub(r"\.(vcf|gz|sqlite)$", "", sqlite_path.name)
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

    Pipeline(med_doc, sqlite_path, config, output_dir, log_level).run()


if __name__ == "__main__":
    app()
