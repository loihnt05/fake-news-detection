import uvicorn
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List

# Import trực tiếp vì 2 file cùng nằm trong folder backend
from verifier import AdvancedFactChecker

# --- GLOBAL VAR ---
checker_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    global checker_instance
    print("🚀 ĐANG KHỞI ĐỘNG SERVER API...")
    try:
        checker_instance = AdvancedFactChecker()
    except Exception as e:
        print(f"❌ LỖI LOAD MODEL: {e}")
    yield
    # SHUTDOWN
    checker_instance = None
    print("🛑 Server đã tắt.")

app = FastAPI(title="Fake News Detection API", lifespan=lifespan)

# --- SCHEMAS ---
class NewsRequest(BaseModel):
    text: str = Field(..., description="Nội dung cần kiểm tra (tiêu đề + nội dung)")

class EvidenceDetail(BaseModel):
    claim: str
    status: str
    score: float
    evidence: str

class VerificationResult(BaseModel):
    status: str
    confidence: float
    explanation: str
    details: List[EvidenceDetail]

# --- ENDPOINTS ---
@app.post("/api/v1/verify", response_model=VerificationResult)
def verify_news(request: NewsRequest):
    if not checker_instance:
        raise HTTPException(status_code=503, detail="Hệ thống đang khởi động...")
    
    try:
        result = checker_instance.verify(request.text)
        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)