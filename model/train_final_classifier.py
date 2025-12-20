import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random

# --- 1. TẠO DỮ LIỆU FEATURE GIẢ LẬP (MÔ PHỎNG PIPELINE) ---
# Vì chạy pipeline thật trên hàng nghìn bài sẽ rất lâu, 
# ta mô phỏng lại các chỉ số mà Pipeline của bạn vừa xuất ra ở trên.

def generate_mock_features(n_samples=1000):
    data = []
    
    # A. Sinh dữ liệu cho bài REAL (Label = 1)
    for _ in range(n_samples // 2):
        # Bài thật thường có:
        # - Có câu được Support (Score > 0.7)
        # - Rất ít câu bị Refute (Score < 0.3)
        # - Avg score cao
        
        n_claims = random.randint(3, 10)
        scores = []
        for _ in range(n_claims):
            # 80% là support, 20% là neutral/noise
            if random.random() < 0.8:
                scores.append(random.uniform(0.75, 0.99)) # High score
            else:
                scores.append(random.uniform(0.4, 0.6))   # Neutral
        
        # Tính feature
        avg_score = np.mean(scores)
        min_score = np.min(scores) # Bài thật thì min score vẫn thường > 0.4
        supported_ratio = sum(1 for s in scores if s > 0.7) / n_claims
        refuted_ratio = sum(1 for s in scores if s < 0.25) / n_claims # Thường là 0
        
        data.append([avg_score, min_score, supported_ratio, refuted_ratio, 1])

    # B. Sinh dữ liệu cho bài FAKE (Label = 0)
    for _ in range(n_samples // 2):
        # Bài giả thường có:
        # - Ít nhất 1 câu bị Refute (Score cực thấp ~0.002)
        # - Avg score thấp
        
        n_claims = random.randint(3, 10)
        scores = []
        # Chắc chắn có 1-2 câu nói điêu
        scores.append(random.uniform(0.001, 0.1)) 
        if random.random() < 0.5: scores.append(random.uniform(0.001, 0.1))
        
        # Còn lại có thể là câu dẫn (neutral) hoặc câu đúng 1 nửa
        for _ in range(n_claims - len(scores)):
            scores.append(random.uniform(0.3, 0.8))
            
        # Tính feature
        avg_score = np.mean(scores)
        min_score = np.min(scores) # Bài giả thì min score cực thấp
        supported_ratio = sum(1 for s in scores if s > 0.7) / n_claims
        refuted_ratio = sum(1 for s in scores if s < 0.25) / n_claims # > 0
        
        data.append([avg_score, min_score, supported_ratio, refuted_ratio, 0])

    df = pd.DataFrame(data, columns=['avg_score', 'min_score', 'supported_ratio', 'refuted_ratio', 'label'])
    return df

# --- 2. TRAIN XGBOOST (Gradient Boosting) ---
def train_classifier():
    print("🛠️ Đang sinh dữ liệu mô phỏng Pipeline...")
    df = generate_mock_features(2000) # 2000 mẫu
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("🚀 Đang train Gradient Boosting Classifier...")
    # XGBoost là thuật toán cực mạnh cho dạng dữ liệu bảng này
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    clf.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = clf.predict(X_test)
    print("\n" + "="*30)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("="*30)
    print(classification_report(y_test, y_pred))
    
    # Lưu model
    joblib.dump(clf, 'final_classifier.pkl')
    print("💾 Đã lưu model tại 'final_classifier.pkl'")

if __name__ == "__main__":
    train_classifier()