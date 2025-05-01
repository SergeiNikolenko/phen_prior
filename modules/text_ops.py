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
        "You are a senior clinical translator. Translate the following text into English. Perform the following steps in strict order "
        "and return ONE final text without any comments, explanations or metadata.\n\n"
        "1. Correct all spelling and grammatical errors, especially in medical terms.\n"
        "2. Identify all abbreviations and replace each with its full expansion, leaving the rest of the text unchanged.\n"
        "3. Remove any information about the patient's mother, father and other relatives; keep only data that relate to the patient.\n"
        "4. Remove all test and investigation results (laboratory values, MRI, CT, etc.); keep only symptoms, complaints and diagnoses.\n"
        "5. Remove all dates and administrative data (chart numbers, addresses, contacts, institution names).\n"
        "6. In the final text keep ONLY:\n"
        "   – the patient's phenotype (age, sex, external appearance if any);\n"
        "   – symptoms and diagnoses;\n"
        "   – test results that are directly related to the diagnosis.\n"
        "   Remove everything else: recommendations, reasoning, references, irrelevant details.\n\n"
        "If a step leaves no corresponding information, simply proceed to the next step; the final text must contain only "
        "the required data without additional explanations."
    )
    processed = chat.ask(text, prompt, temperature=0.1)
    write_text(prompt, "_02_prompt", sample_name, result_dir)
    write_text(processed, "_02_processed_text", sample_name, result_dir)
    return processed
