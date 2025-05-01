# modules/db_ops.py
import sqlite3
import pandas as pd
from pathlib import Path
from .utils import log
from typing import Any

def extract_pos(row: pd.Series, clinprior: pd.DataFrame) -> Any:
    gene = row["Gene"]
    try:
        return clinprior[clinprior["Symbol"] == gene].index[0] + 1
    except IndexError:
        return "None"

def adjust_positions(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]
    df["ACMG"] = pd.Categorical(df["ACMG"], categories=order, ordered=True)
    df = df.sort_values(by=["ACMG", "PositionFunct"])
    df["AdjustedPositionFunct"] = range(1, len(df) + 1)
    return df

def modify_sqlite(sqlite_path: Path, sample_name: str, result_dir: Path):
    csv_path = result_dir / f"{sample_name}_05_clinprior.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ClinPrior CSV not found: {csv_path}")
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute("SELECT base__uid, base__hugo, intervar_new__ACMG FROM variant;").fetchall()
    df = pd.DataFrame(rows, columns=["base__uid", "Gene", "ACMG"])
    clinprior = pd.read_csv(csv_path)
    df["PositionFunct"] = df.apply(lambda r: extract_pos(r, clinprior), axis=1)
    df = adjust_positions(df)
    updates = [(r["AdjustedPositionFunct"], r["base__uid"]) for _, r in df.iterrows()]
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("UPDATE variant SET base__uid = -base__uid WHERE base__uid > 0;")
        conn.executemany("UPDATE variant SET base__uid = ? WHERE base__uid = -?;", updates)
    conn.close()
    log.info("SQLite updated.")
