# modules/utils.py
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich.logging import RichHandler


def setup_logging_file_only(log_file: Path, log_level: str = "info"):
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    fh = logging.FileHandler(log_file, encoding="utf-8")
    selected_level = level_map.get(log_level.lower(), logging.INFO)
    fh.setLevel(selected_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.setLevel(selected_level)
    root_logger.addHandler(fh)
    ch = RichHandler(markup=False, show_time=False, show_level=False, show_path=False)
    ch.setLevel(logging.CRITICAL)
    root_logger.addHandler(ch)


log = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        if not api_key:
            if Path(".env").exists():
                text = Path(".env").read_text()
                if "OPENAI_API_KEY=" in text:
                    api_key = text.split("OPENAI_API_KEY=")[-1].strip()
        if not api_key:
            log.error("OPENAI_API_KEY not provided")
            sys.exit(1)
        self.client = OpenAI(api_key=api_key)

    def ask(
        self,
        text: str,
        prompt: str,
        model: str = "gpt-4.1",
        temperature: float = 0.3,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            stream=False,
            max_tokens=8192,
        )
        return response.choices[0].message.content


def check_file_exists(file_path: Path):
    if not file_path.exists():
        log.error(f"File not found: {file_path}")
        sys.exit(1)


def load_config(config_path: Optional[Path]) -> Optional[str]:
    if config_path is None:
        load_dotenv()
        if Path(".env").exists():
            text = Path(".env").read_text()
            if "OPENAI_API_KEY=" in text:
                return text.split("OPENAI_API_KEY=")[-1].strip()
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        key = cfg.get("openai_api_key")
        if key:
            return key
    except:
        pass
    load_dotenv()
    if Path(".env").exists():
        text = Path(".env").read_text()
        if "OPENAI_API_KEY=" in text:
            return text.split("OPENAI_API_KEY=")[-1].strip()
    return None
