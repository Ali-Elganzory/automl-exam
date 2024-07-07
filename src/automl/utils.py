from enum import Enum
from time import strftime

from torch import distributed as TorchDistributed

DISTRIBUTED = TorchDistributed.is_available() and TorchDistributed.is_initialized()
RANK = 0 if not DISTRIBUTED else TorchDistributed.get_rank()
MAIN_NODE = RANK == 0


class Level(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3


LOG_LEVEL = Level.INFO


def set_log_level(level: Level):
    global LOG_LEVEL
    LOG_LEVEL = level


def log_prefix(level: Level = Level.INFO) -> str:
    return f"{f'[{RANK}] ' if DISTRIBUTED else ''}[{level.name}][{strftime('%Y-%m-%d %H:%M:%S')}]"


def log(
    message: str,
    level: Level = Level.INFO,
    only_main_node: bool = False,
    no_prefix: bool = False,
):
    if only_main_node and not MAIN_NODE:
        return

    if level.value >= LOG_LEVEL.value:
        print(f"{log_prefix(level) if not no_prefix else ''} {message}")
