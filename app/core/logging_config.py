# app/core/logging_config.py
# Centralized logging setup — writes to both the console and a log file.

import logging
import sys
from pathlib import Path

# Ensure the logs/ folder exists (it does, from Phase 2, but this is a safe guard).
Path("logs").mkdir(exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )