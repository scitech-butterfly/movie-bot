from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from infer import ask_question, get_hint
import uuid

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    game_over: bool = False
    session_id: str

class HintResponse(BaseModel):
    hint: str
    session_id: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_movie_question(request: QuestionRequest, http_request: Request):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    session_id = http_request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        answer = ask_question(request.question, session_id)
        resp_lower = answer.lower()
        game_over = resp_lower.startswith("yes") and "correct" in resp_lower
        return {"answer": answer, "game_over": game_over, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hint", response_model=HintResponse)
async def get_movie_hint(http_request: Request):
    session_id = http_request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        hint = get_hint(session_id)
        return {"hint": hint, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
