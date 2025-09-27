from enum import Enum

class Extremeum(Enum):
    MIN = "min"
    MAX = "max"
    
class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"