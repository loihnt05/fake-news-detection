from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time

from model.hybrid_system import FakeNewsDetector


# 1. Khởi tạo App & Load Model
app = FastAPI(
    title="Fake News Detection API",
    description="API phát hiện tin giả sử dụng Hybrid AI (PhoBERT + FAISS)",
    version="1.0"
)

# Biến toàn cục để chứa Model (Load 1 lần duy nhất khi bật server)
detector = None

@app.on_event("startup")
def load_model():
    global detector
    # Khởi tạo model AI
    detector = FakeNewsDetector()
    print("SERVER READY: Model đã load xong!")

# 2. Định nghĩa dữ liệu đầu vào/đầu ra
class NewsRequest(BaseModel):
    text: str # Người dùng gửi lên một chuỗi văn bản

class NewsResponse(BaseModel):
    result: str      # REAL / FAKE / UNDEFINED
    reason: str      # Lý do
    message: str     # Chi tiết
    confidence: float
    process_time: float

# 3. Tạo Endpoint (Cổng kết nối)
@app.post("/check-news", response_model=NewsResponse)
async def check_news(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung bài báo")
    
    # Gọi hàm check của class FakeNewsDetector
    # (Hàm này bạn đã test kỹ ở bước trước)
    result = detector.check(request.text)
    
    # Trả kết quả về cho Extension
    return NewsResponse(
        result=result['result'],
        reason=result['reason'],
        message=result['message'],
        confidence=result['confidence'],
        process_time=result['time']
    )

@app.get("/")
def home():
    return {"status": "running", "message": "Hệ thống AI đang hoạt động!"}

# Chạy server nếu file được gọi trực tiếp
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)