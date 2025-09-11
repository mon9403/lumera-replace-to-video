from pydantic import BaseModel, Field
from typing import Optional

class ComposeResponse(BaseModel):
    composite_url: str
    task_id: Optional[str] = None
    prompt: str

class TaskResponse(BaseModel):
    status: str = Field(description="Kling task status")
    video_url: Optional[str] = None
    detail: Optional[str] = None
