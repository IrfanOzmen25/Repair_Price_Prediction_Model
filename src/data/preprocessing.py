# src/data/preprocessing.py

from __future__ import annotations
import pandas as pd
import numpy as np
from rapidfuzz import process

from src.utils.config import CLEAN_CATEGORIES


def fuzzy_clean_service(raw_name: object, categories=CLEAN_CATEGORIES, threshold: int = 80) -> str:
    name = str(raw_name).upper().strip()
    best_match, score, _ = process.extractOne(name, categories)
    return best_match if score >= threshold else name


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def infer_feature_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # standardize SERVICE
    if "SERVICE" in df.columns:
        df["SERVICE"] = df["SERVICE"].apply(fuzzy_clean_service).astype(str).str.upper().str.strip()
    return df


def fill_missing(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
            df[col] = df[col].fillna(mode_val)

    return df


def filter_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    df = df[df[target].notna()]
    df = df[df[target] >= 0]
    return df
