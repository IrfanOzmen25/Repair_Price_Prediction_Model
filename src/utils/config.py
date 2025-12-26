# src/utils/config.py

from dataclasses import dataclass
from typing import List

CLEAN_CATEGORIES: List[str] = [
    "HAIL",
    "PDR",
    "GLASS",
    "PACKAGE",
    "PPF VINYL PACKAGE",
    "PARTS",
    "WHEEL",
    "TINT",
    "MISC",
    "DETAIL",
    "PPF VINYL",
    "REMOVEINSTALL",
    "PRICE A DENT",
    "PAINT & BODY",
    "DETAIL PACKAGE",
    "PPF VINYL MATERIAL",
]

DEFAULT_CAT_COLS = ["ITEM_NAME", "VEHICLE_MODEL", "REPAIRER_STATE", "SERVICE"]
DEFAULT_ID_LIKE = ["INVOICE_ID", "_BATCH_ID_"]
DEFAULT_TARGET = "ITEM_AMOUNT"

@dataclass(frozen=True)
class TrainConfig:
    target: str = DEFAULT_TARGET
    cat_cols: List[str] = None
    id_like: List[str] = None

    test_size: float = 0.2
    random_state: int = 42

    # winsor/clip (train-only)
    clip_lo: float = 0.005
    clip_hi: float = 0.995

    # model/training
    hidden: int = 128
    dropout: float = 0.2
    batch_train: int = 256
    batch_test: int = 1024
    epochs: int = 20

    lr: float = 1e-3
    weight_decay: float = 1e-4

    # huber (SmoothL1)
    huber_beta: float = 0.5

    # interval from residuals
    interval_lo: float = 0.10
    interval_hi: float = 0.90

    def __post_init__(self):
        object.__setattr__(self, "cat_cols", self.cat_cols or DEFAULT_CAT_COLS)
        object.__setattr__(self, "id_like", self.id_like or DEFAULT_ID_LIKE)
