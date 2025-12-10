"""
Script để test API endpoints sau khi server chạy
Chạy script này sau khi đã start: uvicorn main:app --reload
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("TEST FAKE NEWS DETECTION API")
print("=" * 60)

# 1. Test health check
print("\n1. Testing health check endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("   ✅ Health check OK")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ❌ Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Lỗi: {e}")
    print("   💡 Đảm bảo server đang chạy: uvicorn main:app --reload")
    exit(1)

# 2. Test với tin thật (Real news example)
print("\n2. Testing với tin thật (sample)...")
real_news = {
    "content": "Ngày 10/12/2025, Chính phủ Việt Nam công bố kế hoạch phát triển kinh tế số giai đoạn 2025-2030. Theo đó, mục tiêu đưa kinh tế số đóng góp 30% GDP vào năm 2030.",
    "url": "https://example.com/real-news"
}

try:
    response = requests.post(f"{BASE_URL}/check-news", json=real_news)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ API response OK")
        print(f"   - Status: {result.get('status')}")
        print(f"   - Label: {result.get('label')}")
        print(f"   - Confidence: {result.get('confidence', 0):.3f}")
        print(f"   - Color: {result.get('color')}")
        if 'scores' in result:
            print(f"   - Scores: Real={result['scores']['real']:.3f}, Fake={result['scores']['fake']:.3f}")
    else:
        print(f"   ❌ Status: {response.status_code}")
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Lỗi: {e}")

# 3. Test với tin giả (Fake news example)
print("\n3. Testing với tin giả (sample)...")
fake_news = {
    "content": "KHẨN CẤP: Người ngoài hành tinh đã hạ cánh xuống Hà Nội!!! Chính phủ đang che giấu sự thật này. Chia sẻ ngay để mọi người biết!!!",
    "url": "https://example.com/fake-news"
}

try:
    response = requests.post(f"{BASE_URL}/check-news", json=fake_news)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ API response OK")
        print(f"   - Status: {result.get('status')}")
        print(f"   - Label: {result.get('label')}")
        print(f"   - Confidence: {result.get('confidence', 0):.3f}")
        print(f"   - Color: {result.get('color')}")
        if 'scores' in result:
            print(f"   - Scores: Real={result['scores']['real']:.3f}, Fake={result['scores']['fake']:.3f}")
    else:
        print(f"   ❌ Status: {response.status_code}")
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Lỗi: {e}")

# 4. Test với tin애매 (Ambiguous)
print("\n4. Testing với tin애매한...")
ambiguous_news = {
    "content": "Một số chuyên gia cho rằng giá vàng có thể tăng trong tương lai.",
    "url": "https://example.com/ambiguous"
}

try:
    response = requests.post(f"{BASE_URL}/check-news", json=ambiguous_news)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ API response OK")
        print(f"   - Status: {result.get('status')}")
        print(f"   - Label: {result.get('label')}")
        print(f"   - Confidence: {result.get('confidence', 0):.3f}")
        print(f"   - Color: {result.get('color')}")
        if 'scores' in result:
            print(f"   - Scores: Real={result['scores']['real']:.3f}, Fake={result['scores']['fake']:.3f}")
    else:
        print(f"   ❌ Status: {response.status_code}")
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Lỗi: {e}")

print("\n" + "=" * 60)
print("HOÀN THÀNH TEST")
print("=" * 60)
print("\n💡 Để xem API docs:")
print("   http://localhost:8000/docs")
print("=" * 60)
