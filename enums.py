from enum import Enum


class ExitStatusEnum(Enum):
    """
    Values for the `__status` in `sys.exit()`.
    """

    REGISTER_WORKER_ERROR = 1
