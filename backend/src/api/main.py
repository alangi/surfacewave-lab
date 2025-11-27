import json
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from obspy import read

from swmam.base import get_dispersion_method
from swmam.geometry_io import load_geometry_from_csv
from swmam.timeseries import MamTimeSeries
from swinversion.forward_base import get_forward_solver
from swinversion.model import LayerParamBounds, ModelParametrization
from swinversion.search_base import get_global_search
from swinversion.search_na import NAConfig

from .schemas import (
    FileUploadResponse,
    GeometryUploadResponse,
    InversionRequest,
    JobInfo,
    JobStatusEnum,
    JobStatusResponse,
    JobSubmitResponse,
    MamProcessingRequest,
)

app = FastAPI()

# In-memory job registry. Replace with persistent storage in production.
JOBS: dict[str, JobInfo] = {}
DATA_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = DATA_ROOT / "uploads"
RESULT_ROOT = DATA_ROOT / "results"
GEOMETRY_ROOT = DATA_ROOT / "geometry"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
GEOMETRY_ROOT.mkdir(parents=True, exist_ok=True)


def create_job(job_type: str, initial_status: JobStatusEnum = JobStatusEnum.pending) -> JobInfo:
    job_id = str(uuid.uuid4())
    job = JobInfo(job_id=job_id, job_type=job_type, status=initial_status)
    JOBS[job_id] = job
    return job


def update_job(job_id: str, **fields: Any) -> JobInfo:
    if job_id not in JOBS:
        raise KeyError(f"Job {job_id} not found")
    job = JOBS[job_id]
    job_data = job.model_dump()
    job_data.update(fields)
    updated = JobInfo(**job_data)
    JOBS[job_id] = updated
    return updated


def _is_zip_upload(upload_file: UploadFile) -> bool:
    content_type = (upload_file.content_type or "").lower()
    filename = (upload_file.filename or "").lower()
    return filename.endswith(".zip") or content_type in {"application/zip", "application/x-zip-compressed"}


def load_dataset_stream(dataset_dir: Path):
    valid_exts = {".mseed", ".sgy", ".sg2", ".segy"}
    files = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    if not files:
        raise HTTPException(status_code=400, detail="No supported waveform files found for dataset")
    stream = read([str(p) for p in files])
    if len(stream) == 0:
        raise ValueError("No traces in uploaded dataset")
    return stream


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(files: list[UploadFile] = File(...)) -> FileUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    file_id = str(uuid.uuid4())
    dest_dir = UPLOAD_ROOT / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    is_single_zip = len(files) == 1 and _is_zip_upload(files[0])
    filenames: list[str] = []

    for upload_file in files:
        dest_path = dest_dir / upload_file.filename
        content = await upload_file.read()
        dest_path.write_bytes(content)
        filenames.append(upload_file.filename)

    if is_single_zip:
        zip_path = dest_dir / files[0].filename
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            extracted_files = [info.filename for info in zip_ref.infolist() if not info.is_dir()]
            zip_ref.extractall(dest_dir)
        try:
            zip_path.unlink()
        except OSError:
            pass
        if not extracted_files:
            raise HTTPException(status_code=400, detail="ZIP file contains no files")
        filenames = extracted_files

    return FileUploadResponse(file_id=file_id, filenames=filenames)


@app.post("/geometry/upload", response_model=GeometryUploadResponse)
async def upload_geometry_csv(file: UploadFile = File(...)) -> GeometryUploadResponse:
    """
    Upload a geometry CSV file and return a geometry_id.
    """
    geometry_id = str(uuid.uuid4())
    dest_dir = GEOMETRY_ROOT / geometry_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    content = await file.read()
    dest_path.write_bytes(content)

    return GeometryUploadResponse(geometry_id=geometry_id, filename=file.filename)


@app.post("/mam/dispersion", response_model=JobSubmitResponse)
async def submit_mam_dispersion(
    request: MamProcessingRequest, background_tasks: BackgroundTasks
) -> JobSubmitResponse:
    job = create_job("MAM_PROCESSING")
    background_tasks.add_task(run_mam_job, job.job_id, request)
    return JobSubmitResponse(job_id=job.job_id, status=job.status)


def run_mam_job(job_id: str, request: MamProcessingRequest) -> None:
    try:
        update_job(job_id, status=JobStatusEnum.running)

        dataset_dir = UPLOAD_ROOT / request.file_id
        if not dataset_dir.exists():
            raise FileNotFoundError(f"file_id {request.file_id} not found")
        stream = load_dataset_stream(dataset_dir)

        geom_dir = GEOMETRY_ROOT / request.geometry_id
        if not geom_dir.exists():
            raise FileNotFoundError(f"geometry_id {request.geometry_id} not found")
        geom_files = [p for p in geom_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        if not geom_files:
            raise FileNotFoundError(f"No geometry CSV found for geometry_id {request.geometry_id}")
        geom_path = geom_files[0]

        geom = load_geometry_from_csv(geom_path)

        # Map stream traces to geometry station order
        stream_map = {tr.stats.station: tr for tr in stream}
        missing = [sid for sid in geom.station_ids if sid not in stream_map]
        if missing:
            raise ValueError(f"Stations missing in waveform data: {', '.join(missing)}")

        deltas = [stream_map[sid].stats.delta for sid in geom.station_ids]
        if not all(np.isclose(d, deltas[0]) for d in deltas):
            raise ValueError("Sampling intervals differ across stations; please upload data with consistent sampling")

        n_samples = min(len(stream_map[sid]) for sid in geom.station_ids)
        data = np.stack([stream_map[sid].data[:n_samples] for sid in geom.station_ids]).astype(float)
        data = data.reshape(1, data.shape[0], data.shape[1])

        dt = stream_map[geom.station_ids[0]].stats.delta
        ts = MamTimeSeries(data=data, dt=dt)
        ts.validate()

        method = get_dispersion_method(request.method)
        cfg = request.config
        if hasattr(method, "default_config"):
            cfg_cls = type(getattr(method, "default_config"))
            cfg = cfg_cls(**request.config)

        result = method.estimate(ts, geom, cfg)
        result.validate()

        out_path = RESULT_ROOT / f"{job_id}_mam.json"
        out_path.write_text(json.dumps(result.to_dict()))

        update_job(job_id, status=JobStatusEnum.done, result={"result_path": str(out_path)})
    except Exception as exc:  # pragma: no cover - defensive background task logging
        update_job(job_id, status=JobStatusEnum.error, message=str(exc))


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(job=JOBS[job_id])


@app.post("/inversion", response_model=JobSubmitResponse)
async def submit_inversion(
    request: InversionRequest, background_tasks: BackgroundTasks
) -> JobSubmitResponse:
    job = create_job("INVERSION")
    background_tasks.add_task(run_inversion_job, job.job_id, request)
    return JobSubmitResponse(job_id=job.job_id, status=job.status)


def _load_dispersion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    freq = np.asarray(data["frequency"], dtype=float)
    c = np.asarray(data["phase_velocity"], dtype=float)
    unc = data.get("uncertainty")
    sigma = np.asarray(unc if unc is not None else np.ones_like(c), dtype=float)
    return freq, c, sigma


def _build_param(param_data: dict[str, Any]) -> ModelParametrization:
    layer_bounds_data = param_data.get("layer_bounds", [])
    bounds = [LayerParamBounds(**lb) for lb in layer_bounds_data]
    return ModelParametrization(nlayers=param_data["nlayers"], layer_bounds=bounds)


def run_inversion_job(job_id: str, request: InversionRequest) -> None:
    try:
        update_job(job_id, status=JobStatusEnum.running)

        disp_path = RESULT_ROOT / request.dispersion_file_id
        if not disp_path.exists():
            raise FileNotFoundError(f"dispersion file {disp_path} not found")

        freq, c_obs, sigma_obs = _load_dispersion(disp_path)

        param = _build_param(request.parametrization)
        param.validate()

        solver = get_forward_solver(request.forward_solver)
        search = get_global_search(request.search_method)

        search_cfg = request.search_config
        if search.name.lower() == "na":
            search_cfg = NAConfig(**request.search_config)

        ensemble = search.run(param, freq, c_obs, sigma_obs, solver, search_cfg)
        best_model = ensemble.best_model()
        h, vp, vs, rho = best_model.as_arrays()
        result_data = {
            "best_misfit": ensemble.best_misfit(),
            "best_model": {
                "h": h.tolist(),
                "vp": vp.tolist(),
                "vs": vs.tolist(),
                "rho": rho.tolist(),
            },
            "n_models": len(ensemble.models),
        }

        out_path = RESULT_ROOT / f"{job_id}_inv.json"
        out_path.write_text(json.dumps(result_data))
        update_job(job_id, status=JobStatusEnum.done, result={"result_path": str(out_path)})
    except Exception as exc:  # pragma: no cover - defensive background task logging
        update_job(job_id, status=JobStatusEnum.error, message=str(exc))
