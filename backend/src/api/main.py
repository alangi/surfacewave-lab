import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from obspy import read

from swmam.base import get_dispersion_method
from swmam.geometry import MamArrayGeometry
from swmam.timeseries import MamTimeSeries

from .schemas import (
    FileUploadResponse,
    JobInfo,
    JobStatusEnum,
    JobSubmitResponse,
    MamProcessingRequest,
)

app = FastAPI()

# In-memory job registry. Replace with persistent storage in production.
JOBS: dict[str, JobInfo] = {}
UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploads"
RESULT_ROOT = Path(__file__).resolve().parent.parent / "results"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


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


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
    file_id = str(uuid.uuid4())
    dest_dir = UPLOAD_ROOT / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    content = await file.read()
    dest_path.write_bytes(content)

    return FileUploadResponse(file_id=file_id)


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

        file_dir = UPLOAD_ROOT / request.file_id
        if not file_dir.exists():
            raise FileNotFoundError(f"file_id {request.file_id} not found")
        files = [p for p in file_dir.iterdir() if p.is_file()]
        if not files:
            raise FileNotFoundError(f"No files found for file_id {request.file_id}")
        file_path = files[0]

        stream = read(str(file_path))
        if len(stream) == 0:
            raise ValueError("No traces in uploaded file")

        n_samples = min(len(tr) for tr in stream)
        data = np.stack([tr.data[:n_samples] for tr in stream]).astype(float)
        data = data.reshape(1, data.shape[0], data.shape[1])

        dt = stream[0].stats.delta
        station_ids = [tr.stats.station or f"ST{i}" for i, tr in enumerate(stream)]
        coords = np.zeros((len(station_ids), 2))

        ts = MamTimeSeries(data=data, dt=dt)
        ts.validate()

        geom = MamArrayGeometry(station_ids=station_ids, coords_xy=coords)
        geom.validate()

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
