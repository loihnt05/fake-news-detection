import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import sys
import os
from typing import List, Optional

# --- CẤU HÌNH IMPORT ---
# Thêm đường dẫn để import được class từ thư mục test hoặc thư mục hiện tại
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import class kiểm chứng
try:
    from test.verifier import AdvancedFactChecker
except ImportError:
    try:
        from verifier import AdvancedFactChecker
    except ImportError:
        print("❌ LỖI: Không tìm thấy file verifier.py. Hãy đảm bảo bạn chạy lệnh tại thư mục gốc.")
        sys.exit(1)

# --- CẤU HÌNH LIFESPAN ---
checker_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. STARTUP
    global checker_instance
    print("\n" + "="*60)
    print("🚀 ĐANG KHỞI ĐỘNG SERVER API FACT-CHECKING...")
    print("⏳ Đang load Models (PhoBERT, Bi-Encoder, Cross-Encoder)...")
    try:
        checker_instance = AdvancedFactChecker()
        print("✅ MODEL LOAD THÀNH CÔNG! SẴN SÀNG.")
    except Exception as e:
        print(f"❌ LỖI KHỞI TẠO MODEL: {e}")
        # Không exit app để còn debug được lỗi khác nếu cần, nhưng log rõ ràng
    print("="*60 + "\n")
    
    yield
    
    # 2. SHUTDOWN
    print("🛑 Server đang tắt...")
    checker_instance = None

app = FastAPI(
    title="Vietnamese Fake News Detection API",
    description="API kiểm chứng tin giả sử dụng kiến trúc Neuro-Symbolic (Retrieve-then-Verify).",
    version="1.0.0",
    lifespan=lifespan
)

# --- MODELS (SCHEMA) ---

class NewsRequest(BaseModel):
    title: str = Field(..., examples=["V-League khai mạc tháng 12"])
    content: str = Field(..., examples=["Theo thông tin mới nhất, giải đấu V-League sẽ bắt đầu vào tháng 12 năm nay."])

class EvidenceDetail(BaseModel):
    claim: str
    status: str
    score: float
    evidence: str

class VerificationResult(BaseModel):
    status: str
    confidence: float
    explanation: str
    details: List[EvidenceDetail] # Định nghĩa rõ list chứa gì

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Truy cập /docs để sử dụng giao diện Swagger UI."
    }

@app.post("/api/v1/verify", response_model=VerificationResult)
def verify_news(request: NewsRequest): # Bỏ async để FastAPI dùng threadpool cho tác vụ nặng
    """
    Endpoint chính để kiểm tra tin thật/giả.
    """
    if not checker_instance:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Vui lòng chờ vài giây.")

    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Nội dung không được để trống.")

    # Ghép chuỗi để tăng ngữ cảnh
    full_text = f"{request.title}\n{request.content}"

    try:
        print(f"📩 Nhận request: {request.title[:30]}...")
        result = checker_instance.verify(full_text)
        return result
    except Exception as e:
        print(f"❌ Lỗi xử lý: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Reload=False khi chạy production/model nặng
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)