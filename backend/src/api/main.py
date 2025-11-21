import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile

from .schemas import FileUploadResponse, JobInfo, JobStatusEnum

app = FastAPI()

# In-memory job registry. Replace with persistent storage in production.
JOBS: dict[str, JobInfo] = {}
UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


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
