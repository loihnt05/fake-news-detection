import psycopg2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Cấu hình DB từ .env
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Kiểm tra config
if not DB_CONFIG["dbname"]:
    print("❌ LỖI: Không đọc được file .env. Hãy chắc chắn file .env nằm ở thư mục gốc project.")
    exit(1)

print(f"📊 Kết nối tới database: {DB_CONFIG['dbname']} @ {DB_CONFIG['host']}")

# 1. KẾT NỐI DATABASE LẤY DỮ LIỆU
print("⏳ Đang tải dữ liệu từ PostgreSQL...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # Bảng 'articles', cột vector là 'embedding', cột nhãn là 'label' (0: Fake, 1: Real)
    cur.execute("SELECT embedding, label FROM articles WHERE label IS NOT NULL AND embedding IS NOT NULL")
    rows = cur.fetchall()
    
    # Kiểm tra phân bố nhãn
    cur.execute("SELECT label, COUNT(*) FROM articles WHERE label IS NOT NULL AND embedding IS NOT NULL GROUP BY label ORDER BY label")
    label_counts = cur.fetchall()
    print(f"\n📈 Phân bố dữ liệu:")
    for label, count in label_counts:
        label_name = "Real" if label == 1 else "Fake"
        print(f"   - Label {label} ({label_name}): {count} mẫu")
    
    conn.close()
except Exception as e:
    print(f"❌ Lỗi kết nối hoặc truy vấn database: {e}")
    exit(1)

if len(rows) == 0:
    print("❌ Không tìm thấy dữ liệu có label và embedding. Vui lòng kiểm tra lại database.")
    exit(1)

# Chuyển đổi dữ liệu sang dạng Numpy
# Lưu ý: embedding được lưu dưới dạng string '[0.1, 0.2...]', cần parse ra list
print("\n⏳ Đang chuyển đổi dữ liệu...")
try:
    import ast
    vectors = np.array([np.array(ast.literal_eval(r[0]) if isinstance(r[0], str) else r[0]) for r in rows], dtype=np.float32)
    labels = np.array([r[1] for r in rows], dtype=np.int64)
except Exception as e:
    print(f"❌ Lỗi khi parse embedding: {e}")
    print("💡 Kiểm tra lại format dữ liệu trong cột 'embedding'")
    exit(1)

print(f"✅ Đã tải {len(vectors)} mẫu dữ liệu. Kích thước vector: {vectors.shape[1]}")

# 2. CHUẨN BỊ DỮ LIỆU TRAIN
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)

# Chuyển sang Tensor (Pytorch format)
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. ĐỊNH NGHĨA MODEL (Mạng nơ-ron đơn giản)
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output 2 lớp: Fake (0) và Real (1)
        )
    
    def forward(self, x):
        return self.network(x)

model = NewsClassifier(input_dim=vectors.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. TRAIN MODEL
print("🚀 Bắt đầu train...")
epochs = 10 # Chạy 10 vòng
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 5. LƯU MODEL
torch.save(model.state_dict(), "fakenews_classifier.pth")
print("🎉 Đã train xong và lưu file 'fakenews_classifier.pth'")