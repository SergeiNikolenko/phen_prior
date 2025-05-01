from pathlib import Path
import sys
import json
import logging
import time
from typing import Optional, Callable

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
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
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 90.0,
        max_retries: int = 5,
        backoff: float = 2.0,
    ):
        load_dotenv()
        if not api_key:
            if Path(".env").exists():
                text = Path(".env").read_text()
                if "OPENAI_API_KEY=" in text:
                    api_key = text.split("OPENAI_API_KEY=")[-1].strip()
        if not api_key:
            log.error("OPENAI_API_KEY not provided")
            sys.exit(1)
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.backoff = backoff
        self.tail_cb: Optional[Callable[[str], None]] = None

    def ask(
        self,
        text: str,
        prompt: str,
        model: str = "gpt-4.1",
        temperature: float = 0.3,
    ) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=temperature,
                    stream=True,
                    max_tokens=8192,
                )
                buf = []
                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue
                    buf.append(delta)
                    tail_raw = ("".join(buf))[-100:]
                    tail = tail_raw.replace("\n", " ").replace("\r", " ")
                    if self.tail_cb:
                        self.tail_cb(tail)
                    else:
                        sys.stdout.write("\x1b[2K\r" + tail)
                        sys.stdout.flush()
                if not self.tail_cb:
                    sys.stdout.write("\x1b[2K\r")
                    sys.stdout.flush()
                return "".join(buf).strip()
            except (APIConnectionError, APITimeoutError, RateLimitError) as err:
                log.warning(f"API error: {err.__class__.__name__} – attempt {attempt}/{self.max_retries}")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)

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
    except Exception:
        pass
    load_dotenv()
    if Path(".env").exists():
        text = Path(".env").read_text()
        if "OPENAI_API_KEY=" in text:
            return text.split("OPENAI_API_KEY=")[-1].strip()
    return None
