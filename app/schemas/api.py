from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = "default_session"


class CreateSessionRequest(BaseModel):
    name: str = ""


class RenameSessionRequest(BaseModel):
    name: str