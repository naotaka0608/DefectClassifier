from enum import StrEnum

class TaskType(StrEnum):
    """タスクの種類"""
    CAUSE = "cause"
    SHAPE = "shape"
    DEPTH = "depth"
