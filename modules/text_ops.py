# modules/text_ops.py
import re
import nltk
import transformers
from pathlib import Path
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

def translate_text(text: str, chat: DeepSeekClient) -> str:
    clean = re.sub(r"[^A-Za-zА-Яа-я.:;,-?!()0-9]", " ", text)
    prompt = (
        "Переведи следующий текст медицинской направленности на английский язык, сохраняя точность терминологии и стиля. "
        "При переводе убери все переносы строк, верни сплошной текст."
    )
    text_eng = chat.ask(clean, prompt, temperature=1.3)
    return re.sub(r"\n", " ", text_eng)

def process_text(text: str, chat: DeepSeekClient, sample_name: str, result_dir: Path) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > 8192:
        p1, p2 = split_text(text)
        return process_text(p1, chat, sample_name, result_dir) + "\n" + process_text(p2, chat, sample_name, result_dir)
    prompt = (
        "Исправь все опечатки в следующем тексте. Убедись, что текст не содержит грамматических "
        "или орфографических ошибок, особенно в терминах медицинской направленности. "
        "Верни только исправленный текст без дополнительных комментариев или пояснений."
    )
    processed = chat.ask(text, prompt, temperature=0.0)
    prompt = (
        "Ты должен:\n1. Найти все аббревиатуры в предоставленном медицинском тексте\n"
        "2. Заменить каждую аббревиатуру на её полную расшифровку\n"
        "3. Сохранить исходный текст, изменив только аббревиатуры\n"
        "4. Если аббревиатур нет - вернуть исходный текст"
    )
    processed = chat.ask(processed, prompt, temperature=0.3)
    prompt = (
        "Ты должен:\n1. Удалить из текста ВСЮ информацию о матери, отце и других родственниках\n"
        "2. Сохранить только данные, относящиеся непосредственно к пациенту\n"
        "3. Если информации о родственниках нет — вернуть исходный текст"
    )
    processed = chat.ask(processed, prompt, temperature=1.0)
    prompt = (
        "Ты должен:\n1. Удалить из текста ВСЕ упоминания результатов анализов\n"
        "2. Оставить только симптомы, жалобы и диагнозы\n"
        "3. Если анализов нет — вернуть текст без изменений"
    )
    processed = chat.ask(processed, prompt, temperature=1.0)
    prompt = (
        "Ты должен:\n1. Удалить из текста ВСЕ даты\n2. Удалить административные данные\n"
        "3. Оставить только медицинскую информацию\n4. Если таких данных нет — вернуть исходный текст"
    )
    processed = chat.ask(processed, prompt, temperature=1.0)
    prompt = (
        "Ты должен:\n1. Извлечь из текста ТОЛЬКО информацию о фенотипе, симптомах, диагнозах\n"
        "2. Удалить всё остальное\n"
        "3. Если текст уже содержит только нужные данные — вернуть его без изменений"
    )
    processed = chat.ask(processed, prompt, temperature=1.0)
    write_text(processed, "_processed_text", sample_name, result_dir)
    return processed
