"""Logging configuration for the PathSnap API service."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_api_logger(logs_dir: str | Path = "experiments/logs") -> logging.Logger:
    """Set up and configure the API logger with console and file handlers."""
    log_dir_path = Path(logs_dir)
    if not log_dir_path.is_absolute():          # relative -> repo root (works from any cwd)
        log_dir_path = Path(__file__).resolve().parents[1] / log_dir_path
    log_dir_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    log_file_path = log_dir_path / f"api_{date_str}.log"

    logger = logging.getLogger("pathsnap.api")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if reloaded
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
