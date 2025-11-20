import numpy as np
import numpy.testing as npt

from swmam.dispersion import MamDispersionResult
from swmam.geometry import MamArrayGeometry
from swmam.timeseries import MamTimeSeries
from swinversion.ensemble import ModelEnsemble
from swinversion.model import LayerParamBounds, ModelParametrization, VsLayer, VsModel


def test_mam_geometry_and_timeseries_validation():
    station_ids = ["STA1", "STA2", "STA3"]
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    elevations = np.array([100.0, 101.0, 99.5])

    geom = MamArrayGeometry(station_ids=station_ids, coords_xy=coords, elevations=elevations)
    geom.validate()

    data = np.ones((2, 3, 5))
    ts = MamTimeSeries(data=data, dt=0.01)
    ts.validate()
    assert ts.n_components == 2
    assert ts.n_stations == 3
    assert ts.n_samples == 5


def test_dispersion_to_from_dict_and_validation():
    freq = np.array([1.0, 2.0, 3.0])
    c = np.array([200.0, 250.0, 300.0])
    unc = np.array([5.0, 5.0, 5.0])

    disp = MamDispersionResult(method="freq-bessel", frequency=freq, phase_velocity=c, uncertainty=unc)
    disp.validate()

    disp_dict = disp.to_dict()
    restored = MamDispersionResult.from_dict(disp_dict)
    restored.validate()

    npt.assert_allclose(restored.frequency, freq)
    npt.assert_allclose(restored.phase_velocity, c)
    npt.assert_allclose(restored.uncertainty, unc)


def test_model_parametrization_and_model_round_trip():
    bounds = [
        LayerParamBounds(vs_min=100.0, vs_max=200.0, h_min=1.0, h_max=5.0),
        LayerParamBounds(vs_min=200.0, vs_max=400.0, h_min=5.0, h_max=10.0),
    ]
    param = ModelParametrization(nlayers=2, layer_bounds=bounds)
    param.validate()

    h = np.array([3.0, 7.0])
    vp = np.array([500.0, 700.0])
    vs = np.array([150.0, 300.0])
    rho = np.array([1900.0, 2100.0])

    model = VsModel.from_arrays(h=h, vp=vp, vs=vs, rho=rho)
    h_rt, vp_rt, vs_rt, rho_rt = model.as_arrays()

    npt.assert_allclose(h_rt, h)
    npt.assert_allclose(vp_rt, vp)
    npt.assert_allclose(vs_rt, vs)
    npt.assert_allclose(rho_rt, rho)


def test_model_ensemble_best_selection():
    model_a = VsModel(layers=[VsLayer(vs=150.0, h=3.0, vp=500.0, rho=1900.0)])
    model_b = VsModel(layers=[VsLayer(vs=200.0, h=4.0, vp=550.0, rho=1950.0)])

    ensemble = ModelEnsemble(models=[model_a, model_b], misfits=np.array([1.0, 0.4]))

    assert ensemble.best_index() == 1
    assert ensemble.best_model() is model_b
    assert ensemble.best_misfit() == 0.4
