"""The nested Dirichlet prior now lives in ``mcda_studio.methods.nested_dirichlet``.

It moved so that the conviction prior can be drawn by the application without importing the
calibration package (the outcome model stays offline). Names are re-exported here unchanged.
"""

from mcda_studio.methods.nested_dirichlet import *  # noqa: F403
from mcda_studio.methods.nested_dirichlet import __all__ as __all__
