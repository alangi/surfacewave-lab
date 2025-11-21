import json

import numpy as np

from swmam.dispersion import MamDispersionResult


def test_dispersion_result_to_dict_json_safe():
    result = MamDispersionResult(
        method="test",
        frequency=np.array([1.0, 2.0]),
        phase_velocity=np.array([200.0, 250.0]),
        uncertainty=np.array([5.0, 6.0]),
        meta={"radii": np.array([10.0, 20.0]), "nested": {"vals": np.array([1, 2])}},
    )

    payload = result.to_dict()
    # Should be JSON-serializable without errors
    serialized = json.dumps(payload)
    assert serialized

    restored = MamDispersionResult.from_dict(payload)
    restored.validate()
    assert np.allclose(restored.frequency, result.frequency)
    assert np.allclose(restored.phase_velocity, result.phase_velocity)
    assert np.allclose(restored.uncertainty, result.uncertainty)
    # meta entries should be plain lists after round-trip
    assert restored.meta["radii"] == [10.0, 20.0]
