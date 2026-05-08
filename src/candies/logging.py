import os
import sys
import orjson
import logging
import structlog


def isipython() -> bool:
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False


shared_processors = [
    structlog.processors.add_log_level,
    structlog.contextvars.merge_contextvars,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]

if sys.stderr.isatty() or isipython():
    processors = shared_processors + [structlog.dev.ConsoleRenderer()]
else:
    processors = shared_processors + [
        structlog.processors.format_exc_info,
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(serializer=orjson.dumps),
    ]

if not structlog.is_configured():
    structlog.contextvars.clear_contextvars()
    structlog.configure(
        processors=processors,
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(
            {
                "0": logging.INFO,
                "1": logging.DEBUG,
                "True": logging.DEBUG,
                "False": logging.INFO,
            }[str(os.environ.get("CANDIES_DEBUG", "False"))]
        ),
        logger_factory=(
            structlog.BytesLoggerFactory()
            if not (sys.stderr.isatty() or isipython())
            else structlog.PrintLoggerFactory(sys.stdout)
        ),
    )
log = structlog.get_logger()

__all__ = ["log"]
