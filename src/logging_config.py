import logging
import sys


def setup_logging(
    level: str = "INFO",
    log_file: str = "app.log",
):
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # stderr (для MCP)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    # file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(stderr_handler)
    root.addHandler(file_handler)
