import os
import re
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

folder_path = "/Users/nikolenko/Documents/Project/LigandPro/work/mipt/rotation_nikolenko/med_docs/EVG00000216"
file_names = [
    "Заключение Пузняк И.А..txt",
    "Консультация невролога-эпилептолога.txt",
    "рез.исследования_мама.txt",
    "рез.исследования_отец.txt",
    "рез.исследования_пробанд.txt",
    "Закл.рез.исследования_пробанд.txt"
]

texts = []
for name in file_names:
    with open(os.path.join(folder_path, name), "r", encoding="utf-8") as f:
        texts.append(f.read())
combined_text = "\n".join(texts)


months = [
    "январь", "февраль", "март", "апрель", "май",
    "июнь", "июль", "август", "сентябрь", "октябрь",
    "ноябрь", "декабрь"
]
month_forms = [
    (m[:-1] if m.endswith(("ь", "й")) else m) + suffix
    for m in months
    for suffix in ("я", "е")
]

config = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}]
}
nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["ru"])
results = analyzer.analyze(text=combined_text, language="ru")

anonymizer = AnonymizerEngine()
text = anonymizer.anonymize(text=combined_text, analyzer_results=results).text

pattern_months = r"\b(?:" + "|".join(month_forms) + r")(?:-(?:" + "|".join(month_forms) + r"))*\b"
text = re.sub(pattern_months, "<MONTH>", text, flags=re.IGNORECASE)

pattern_years = r"\b(?:19\d{2}|20[0-4]\d|2050)\b"
text = re.sub(pattern_years, "<YEAR>", text)

output_path = os.path.join(folder_path, "EVG00000216_combined.txt")
with open(output_path, "w", encoding="utf-8") as out:
    out.write(text)

print("Anonymization and cleaning completed. Output file:", output_path)