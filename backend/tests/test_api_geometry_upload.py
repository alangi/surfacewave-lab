from fastapi.testclient import TestClient

from api.main import app, GEOMETRY_ROOT


def test_geometry_upload_endpoint(tmp_path, monkeypatch):
    client = TestClient(app)

    # Isolate geometry storage to temp dir for the test
    monkeypatch.setattr("api.main.GEOMETRY_ROOT", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    files = {"file": ("geom.csv", b"Name,Easting,Northing,Elevation\nS1,1,2,3\n")}
    resp = client.post("/geometry/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "geometry_id" in data and data["geometry_id"]
    assert data["filename"] == "geom.csv"

    geom_dir = tmp_path / data["geometry_id"]
    assert geom_dir.exists()
