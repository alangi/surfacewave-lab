import numpy as np

from swmam.base import get_dispersion_method
from swmam.geometry import MamArrayGeometry
from swmam.timeseries import MamTimeSeries
import swmam.hfk as _hfk  # noqa: F401  # ensure registration


def test_hfk_estimate_runs_and_shapes():
    method = get_dispersion_method("hfk")

    # Linear array along x-axis
    station_ids = ["S1", "S2", "S3", "S4"]
    spacing = 10.0
    coords = np.column_stack((np.arange(len(station_ids)) * spacing, np.zeros(len(station_ids))))
    geom = MamArrayGeometry(station_ids=station_ids, coords_xy=coords)
    geom.validate()

    # Plane wave at single frequency with velocity 200 m/s, azimuth 0 deg
    f0 = 2.0
    v_true = 200.0
    dt = 0.01
    n_samples = 1024
    t = np.arange(n_samples) * dt
    phase_shifts = 2 * np.pi * f0 * coords[:, 0] / v_true
    signal = np.sin(2 * np.pi * f0 * t)

    n_comp = 3
    data = np.zeros((n_comp, len(station_ids), n_samples))
    for i, phi in enumerate(phase_shifts):
        data[0, i] = np.sin(2 * np.pi * f0 * t - phi)
    ts = MamTimeSeries(data=data, dt=dt)
    ts.validate()

    cfg = method.default_config
    result = method.estimate(ts, geom, cfg)

    assert result.frequency.shape == result.phase_velocity.shape
    assert np.all(np.isfinite(result.phase_velocity))
