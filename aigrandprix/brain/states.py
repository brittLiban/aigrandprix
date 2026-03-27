"""DroneState enum — the five states of the FusionBrain state machine."""
from enum import Enum, auto


class DroneState(Enum):
    SEARCH = auto()    # no gate visible; yaw-scanning
    TRACK = auto()     # gate detected but not yet aligned
    APPROACH = auto()  # aligned; flying toward gate
    COMMIT = auto()    # close enough; driving through regardless of flicker
    RECOVER = auto()   # instability or long gate loss; stabilise first
