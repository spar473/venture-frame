"""The ``weight-samples`` sidecar model now lives in ``mcda_studio.domain.weight_samples``.

It moved so that SMAA-2 and the application can read a sidecar without importing the calibration
package. Names are re-exported here unchanged.
"""

from mcda_studio.domain.weight_samples import *  # noqa: F403
from mcda_studio.domain.weight_samples import __all__ as __all__
