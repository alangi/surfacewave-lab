from pathlib import Path

from swmam.geometry_io import load_geometry_from_csv


def test_load_geometry_from_csv(tmp_path: Path):
    sample = tmp_path / "geom.csv"
    sample.write_text(
        "Name,Easting,Northing,Elevation\n"
        "STA1,100.0,200.0,10.0\n"
        "STA2,110.0,210.0,11.0\n"
    )

    geom = load_geometry_from_csv(sample)

    n = len(geom.station_ids)
    assert n == 2
    assert geom.coords_xy.shape == (n, 2)
    assert geom.elevations.shape == (n,)
    assert geom.station_ids == ["STA1", "STA2"]
