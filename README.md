# Phenotype prioritizer

## Description

Prioritizations includes the following steps:
1. Parsing of the medical text with DeepSeek API to extract relevenat information about patients phenotype
2. PhenoTagger execution for HPO terms extraction
3. ClinPrior execution to perform gene-level prioritization based on HPO-terms
4. Modification of SQLite file base__uid column in order to sort variants the way they were prioritized

## Dependencies
```

```
* aschluterclinprior/clinprior2
* albertea/phenotagger:1.2
* karchinlab/opencravat:2.8.0

## Execution
```
uv run main.py
```
