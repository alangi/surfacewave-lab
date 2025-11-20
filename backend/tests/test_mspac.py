import numpy as np

from swmam.base import get_dispersion_method
from swmam.geometry import MamArrayGeometry
from swmam.timeseries import MamTimeSeries


def test_mspac_estimate_runs_and_shapes():
    method = get_dispersion_method("mspac")
    # Simple geometry with three stations forming a triangle.
    geom = MamArrayGeometry(
        station_ids=["A", "B", "C"],
        coords_xy=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
    )
    geom.validate()

    # Random vertical-only data (other components zeros) for 3 stations.
    n_comp, n_stations, n_samples = 3, 3, 128
    data = np.zeros((n_comp, n_stations, n_samples))
    rng = np.random.default_rng(0)
    data[0] = rng.standard_normal((n_stations, n_samples))
    ts = MamTimeSeries(data=data, dt=0.01)
    ts.validate()

    config = method.default_config
    result = method.estimate(ts, geom, config)

    assert result.frequency.shape == result.phase_velocity.shape
    assert result.frequency.ndim == 1
