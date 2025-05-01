# modules/text_ops.py
import re
from pathlib import Path
import nltk
import transformers
from .utils import log, DeepSeekClient

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / "data"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    str(TOKENIZER_DIR),
    trust_remote_code=True,
)

def split_text(text: str):
    sentences = nltk.sent_tokenize(text)
    mid = len(sentences) // 2
    return " ".join(sentences[:mid]), " ".join(sentences[mid:])

def write_text(text: str, suffix: str, sample_name: str, result_dir: Path):
    path = result_dir / f"{sample_name}{suffix}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    log.debug(f"Text successfully written: {path}")

def process_text(text: str, chat: DeepSeekClient, sample_name: str, result_dir: Path) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > 8192:
        p1, p2 = split_text(text)
        return process_text(p1, chat, sample_name, result_dir) + "\n" + process_text(p2, chat, sample_name, result_dir)
    prompt = (
        "You are a senior clinical translator. Receive Russian medical text and convert it into a fluent English "
        "clinical narrative suitable for automated phenotypic annotation by PhenoTagger (PubTator).\n\n"
        "Output REQUIREMENTS:\n"
        "- Return a continuous paragraph (or several full sentences) of plain text, no bullet points, no numbering.\n"
        "- Do NOT append HPO codes, summaries or any extra commentary.\n\n"
        "Steps you must follow:\n"
        "1. Translate the entire text to English, preserving accurate medical terminology.\n"
        "2. Expand every Russian or Latin abbreviation to its full form.\n"
        "3. Correct all spelling and grammar errors.\n"
        "4. Remove personal identifiers (names, addresses, record numbers, hospital names) and specific calendar dates; keep relative durations (e.g. “for three months”).\n"
        "5. KEEP every clinically relevant statement about the patient: symptoms, signs, diagnoses, procedures, anatomical descriptions, and observable findings.\n"
        "6. Remove laboratory numeric values, medication lists, treatment recommendations and administrative details.\n"
        "7. Preserve explicit negations (e.g. \"no fever\", \"no seizures\") in the same sentence—they improve downstream NER.\n\n"
        "Return ONLY the cleaned English clinical narrative text."
    )
    processed = chat.ask(text, prompt, temperature=0.1)
    write_text(prompt, "_02_prompt", sample_name, result_dir)
    write_text(processed, "_02_processed_text", sample_name, result_dir)
    return processed
