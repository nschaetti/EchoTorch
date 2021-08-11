

# Imports
import echotorch.datasets.functional as F
from echotorch.viz import timescatter


# Henon strange attractor
henon_series = F.henon(
    size=1,
    length=100,
    xy=(0, 0),
    a=1.4,
    b=0.3,
    washout=0
)

# Show points
timescatter(henon_series[0])
