from pathlib import Path
import re
import logging
import typer
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
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
from modules.utils import setup_logging_file_only, log

app = typer.Typer(add_help_option=False)

MONTH_RE = (
    r"(?:январ(?:[ьяе])|феврал(?:[ьяе])|март(?:[ьае])|апрел(?:[ьяе])|май|июн(?:[ьяе])|"
    r"июл(?:[ьяе])|август(?:[ьае])|сентябр(?:[ьяе])|октябр(?:[ьяе])|ноябр(?:[ьяе])|"
    r"декабр(?:[ьяе]))(?:-(?:январ(?:[ьяе])|феврал(?:[ьяе])|март(?:[ьае])|апрел(?:[ьяе])|май|"
    r"июн(?:[ьяе])|июл(?:[ьяе])|август(?:[ьае])|сентябр(?:[ьяе])|октябр(?:[ьяе])|ноябр(?:[ьяе])|"
    r"декабр(?:[ьяе])))*"
)
YEAR_RE = r"\b(?:19\d{2}|20[0-4]\d|2050)\b"


def _silence_external_logs():
    noisy = [
        "presidio",
        "presidio-analyzer",
        "spacy",
        "pymorphy3",
        "urllib3",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]


def anonymize_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine) -> str:
    res = analyzer.analyze(text=text, language="ru")
    txt = anonymizer.anonymize(text=text, analyzer_results=res).text
    txt = re.sub(MONTH_RE, "<MONTH>", txt, flags=re.IGNORECASE)
    return re.sub(YEAR_RE, "<YEAR>", txt)


def preprocess_folder(
    src: Path,
    dst: Path,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
):
    txt_files = sorted(p for p in src.glob("*.txt") if not p.name.endswith("_combined.txt"))
    combined = "\n".join(p.read_text(encoding="utf-8") for p in txt_files)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(anonymize_text(combined, analyzer, anonymizer), encoding="utf-8")
    log.info(f"Output written: {dst}")


@app.command()
def run(
    base_dir: Path = typer.Option(
        "med_docs",
        "-i",
        "--input",
        help="Базовый каталог с подпапками-источниками",
    ),
    results_dir: Path = typer.Option(
        "results",
        "-o",
        "--output",
        help="Каталог для сохранения объединённых файлов",
    ),
    log_level: str = typer.Option("info", "--log-level", help="Уровень логирования в файл"),
):
    base_dir = base_dir.resolve()
    if not base_dir.exists():
        typer.echo(f"Input directory not found: {base_dir}")
        raise typer.Exit(code=1)

    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging_file_only(results_dir / "batch_preprocess.log", log_level)
    _silence_external_logs()

    cfg = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}],
    }
    nlp = NlpEngineProvider(nlp_configuration=cfg).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["ru"])
    anonymizer = AnonymizerEngine()

    folders = [p for p in base_dir.iterdir() if p.is_dir()]
    with Progress(
        SpinnerColumn(),
        TextColumn("Processing folders:"),
        BarColumn(bar_width=None),
        TaskProgressColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(),
    ) as bar:
        task = bar.add_task("Batch", total=len(folders))
        for sub in folders:
            preprocess_folder(
                sub,
                results_dir / f"{sub.name}_combined.txt",
                analyzer,
                anonymizer,
            )
            bar.update(task, advance=1)


if __name__ == "__main__":
    app()
