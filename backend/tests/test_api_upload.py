import io
import zipfile

from fastapi.testclient import TestClient

from api import main


def _patch_upload_root(tmp_path, monkeypatch):
    monkeypatch.setattr("api.main.UPLOAD_ROOT", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)


def test_upload_single_file_still_works(tmp_path, monkeypatch):
    client = TestClient(main.app)
    _patch_upload_root(tmp_path, monkeypatch)

    files = [("files", ("dummy.mseed", b"waveform", "application/octet-stream"))]
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["filenames"] == ["dummy.mseed"]
    assert data["file_id"]

    dataset_dir = tmp_path / data["file_id"]
    assert (dataset_dir / "dummy.mseed").read_bytes() == b"waveform"


def test_upload_multiple_files_creates_single_dataset(tmp_path, monkeypatch):
    client = TestClient(main.app)
    _patch_upload_root(tmp_path, monkeypatch)

    files = [
        ("files", ("a.mseed", b"a", "application/octet-stream")),
        ("files", ("b.mseed", b"b", "application/octet-stream")),
        ("files", ("c.mseed", b"c", "application/octet-stream")),
    ]
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_id"]
    assert data["filenames"] == ["a.mseed", "b.mseed", "c.mseed"]

    dataset_dir = tmp_path / data["file_id"]
    for name, content in [("a.mseed", b"a"), ("b.mseed", b"b"), ("c.mseed", b"c")]:
        assert (dataset_dir / name).read_bytes() == content


def test_upload_zip_is_extracted(tmp_path, monkeypatch):
    client = TestClient(main.app)
    _patch_upload_root(tmp_path, monkeypatch)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("x.mseed", b"x")
        zf.writestr("y.mseed", b"y")
    buffer.seek(0)

    files = [("files", ("data.zip", buffer.getvalue(), "application/zip"))]
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_id"]
    assert "x.mseed" in data["filenames"]
    assert "y.mseed" in data["filenames"]

    dataset_dir = tmp_path / data["file_id"]
    assert (dataset_dir / "x.mseed").read_bytes() == b"x"
    assert (dataset_dir / "y.mseed").read_bytes() == b"y"
