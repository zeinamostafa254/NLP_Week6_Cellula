"""
Centralized logging setup. Import `setup_logging()` once at app startup
(main.py / streamlit entrypoint) so every module's `logging.getLogger(...)`
writes to the same place with the same format.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from .config import LOGGING


def setup_logging() -> None:
    os.makedirs(os.path.dirname(LOGGING.file) or ".", exist_ok=True)

    root = logging.getLogger("evalgen")
    root.setLevel(LOGGING.level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if root.handlers:
        return  # avoid duplicate handlers on reload (e.g. Streamlit re-runs)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOGGING.file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
