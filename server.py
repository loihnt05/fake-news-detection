from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# Add test directory to path to import verifier module
sys.path.insert(0, str(Path(__file__).parent / "test"))
from verifier import NewsVerifier

import uvicorn

# 1. Khởi tạo App
app = FastAPI(
    title="Fake News Detection API",
    description="Hệ thống kiểm tra tin giả dựa trên Database VNExpress + AI Hybrid Search",
    version="1.0"
)

# 2. Load Model AI (Chỉ load 1 lần khi server start)
# Biến global để lưu instance của verifier
checker = None

@app.on_event("startup")
def load_models():
    global checker
    print("⏳ Đang khởi động Server và Load Model AI...")
    checker = NewsVerifier()
    print("✅ Server đã sẵn sàng!")

# 3. Định nghĩa Data Input/Output (Pydantic)
class NewsRequest(BaseModel):
    title: str
    content: str

class VerificationResult(BaseModel):
    status: str          # TRUE / FAKE / UNDEFINED / ERROR
    explanation: str     # Lý do ngắn gọn
    source_title: str | None = None
    source_url: str | None = None
    details: list | None = None

# 4. Tạo Endpoint API
@app.post("/check", response_model=VerificationResult)
def check_news(request: NewsRequest):
    if not checker:
        raise HTTPException(status_code=503, detail="Server chưa khởi động xong AI Model")
    
    try:
        # Gọi hàm verify trong core logic cũ
        result = checker.verify(request.title, request.content)
        
        # Chuẩn hóa dữ liệu trả về cho đẹp
        response = VerificationResult(
            status=result["status"],
            explanation=result["explanation"],
            source_title=result.get("source_title"),
            source_url=result.get("source_url"),
            details=result.get("details")
        )
        return response

    except Exception as e:
        print(f"❌ Lỗi server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chạy server nếu file được gọi trực tiếp
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)