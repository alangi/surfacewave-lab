from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class JobStatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobInfo(BaseModel):
    job_id: str
    job_type: str
    status: JobStatusEnum
    message: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatusEnum


class JobStatusResponse(BaseModel):
    job: JobInfo


class FileUploadResponse(BaseModel):
    file_id: str
    filenames: list[str]


class MamProcessingRequest(BaseModel):
    file_id: str
    method: str
    config: dict[str, Any]
    geometry_id: str  # reference to uploaded geometry CSV


class GeometryUploadResponse(BaseModel):
    geometry_id: str
    filename: str


class InversionRequest(BaseModel):
    dispersion_file_id: str
    search_method: str
    forward_solver: str
    parametrization: dict[str, Any]
    search_config: dict[str, Any]
