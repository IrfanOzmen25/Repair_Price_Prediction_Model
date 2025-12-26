# src/data/feature_engineering.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_vocab(series: pd.Series, min_freq: int = 2) -> dict[str, int]:
    vc = series.astype(str).value_counts()
    tokens = vc[vc >= min_freq].index.tolist()
    # 0 reserved for UNK
    stoi = {tok: i + 1 for i, tok in enumerate(tokens)}
    return stoi


def encode_cats(frame: pd.DataFrame, vocabs: dict[str, dict[str, int]], cat_cols: list[str]) -> np.ndarray:
    cols = []
    for col in cat_cols:
        stoi = vocabs[col]
        ids = frame[col].astype(str).map(lambda x: stoi.get(x, 0)).astype(np.int64).values
        cols.append(ids)
    return np.stack(cols, axis=1)  # [N, num_cat_cols]


def emb_dim(cardinality: int) -> int:
    return int(min(50, round(np.sqrt(cardinality))))


def split_train_test(df: pd.DataFrame, test_size: float, random_state: int):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def train_only_clip_target(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, lo: float, hi: float):
    low_q, high_q = train_df[target].quantile([lo, hi])
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[target] = train_df[target].clip(lower=low_q, upper=high_q)
    test_df[target] = test_df[target].clip(lower=low_q, upper=high_q)
    return train_df, test_df, float(low_q), float(high_q)


def build_num_cols(df: pd.DataFrame, target: str, id_like: list[str]) -> list[str]:
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    num_cols = [c for c in num_cols if c not in set(id_like)]
    return num_cols


def scale_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[num_cols].astype(float))
    X_test = scaler.transform(test_df[num_cols].astype(float))
    return scaler, X_train, X_test


def build_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str], min_freq: int = 2):
    vocabs = {col: build_vocab(train_df[col], min_freq=min_freq) for col in cat_cols}
    X_cat_train = encode_cats(train_df, vocabs, cat_cols)
    X_cat_test = encode_cats(test_df, vocabs, cat_cols)
    cat_cardinalities = {col: len(vocabs[col]) + 1 for col in cat_cols}  # +1 for UNK=0
    emb_dims = {col: emb_dim(cat_cardinalities[col]) for col in cat_cols}
    return vocabs, X_cat_train, X_cat_test, cat_cardinalities, emb_dims
