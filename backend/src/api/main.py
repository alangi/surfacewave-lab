import uuid
from typing import Any

from .schemas import JobInfo, JobStatusEnum

# In-memory job registry. Replace with persistent storage in production.
JOBS: dict[str, JobInfo] = {}


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

