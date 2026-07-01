# FastAPI app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import ask

app = FastAPI(
    title="Supply Chain RAG Assistant",
    description="AI-powered Q&A for Apple EMEIA Supply Chain operations using RAG",
    version="1.0.0"
)


class QuestionRequestDto(BaseModel):
    question: str


class AnswerResponseDto(BaseModel):
    question: str
    answer: str


@app.get("/")
def root():
    return {"message": "Supply Chain RAG Assistant is running!"}


@app.post("/ask", response_model=AnswerResponseDto)
def ask_question(request: QuestionRequestDto):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = ask(request.question)
    return AnswerResponseDto(question=request.question, answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
