import sys

from loguru import logger

from .config import get_settings


def configure_logging() -> None:
    settings = get_settings()

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        "logs/app.log",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        serialize=True,  # JSON structured logs for ingestion
    )
