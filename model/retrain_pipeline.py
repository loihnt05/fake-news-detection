"""
Retrain Pipeline cho Fact-Check AI
Được gọi bởi Airflow DAG hàng tuần

Pipeline:
1. Lấy dữ liệu training mới từ user feedback đã được admin duyệt
2. Chuẩn bị dữ liệu theo format NLI (claim, evidence, label)
3. Fine-tune Cross-Encoder model
4. Evaluate và lưu metrics
5. Lưu model mới

Usage:
    python model/retrain_pipeline.py
"""

import os
import sys
import json
import torch
import psycopg2
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

# Model paths
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "my_model_v7")
OUTPUT_DIR = Path("model/retrained_models")
MIN_TRAINING_SAMPLES = 50  # Số lượng tối thiểu để retrain

# Training hyperparameters
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100

class RetrainPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conn = None
        self.new_version = None
        
    def connect_db(self):
        """Kết nối database"""
        self.conn = psycopg2.connect(**DB_CONFIG)
        print(f"✅ Connected to DB: {DB_CONFIG['dbname']}")
        
    def get_new_training_data(self):
        """
        Lấy dữ liệu training mới từ:
        1. User reports đã được admin APPROVED
        2. Chưa được dùng để train version nào
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    td.id,
                    td.claim_text,
                    td.evidence_text,
                    td.label
                FROM training_data td
                WHERE td.used_in_version IS NULL
                ORDER BY td.created_at ASC
            """)
            rows = cur.fetchall()
            
        data = []
        for row in rows:
            # Convert NLI label to numeric
            label_map = {
                'ENTAILMENT': 1,    # SUPPORTED
                'CONTRADICTION': 0, # REFUTED
                'NEUTRAL': 2        # NEI
            }
            
            data.append({
                'id': row[0],
                'claim': row[1],
                'evidence': row[2],
                'label': label_map.get(row[3], 2)
            })
            
        return data
    
    def prepare_approved_reports(self):
        """
        Chuyển đổi approved reports thành training data
        Logic:
        - User báo FAKE + AI nói REAL → CONTRADICTION
        - User báo REAL + AI nói FAKE → ENTAILMENT
        """
        with self.conn.cursor() as cur:
            # Lấy reports đã approved chưa được xử lý
            cur.execute("""
                SELECT 
                    r.id,
                    c.content as claim_text,
                    r.user_feedback,
                    r.ai_label_at_report
                FROM user_reports r
                JOIN claims c ON r.claim_id = c.id
                WHERE r.status = 'APPROVED'
                AND r.id NOT IN (SELECT report_id FROM training_data WHERE report_id IS NOT NULL)
            """)
            
            rows = cur.fetchall()
            new_samples = 0
            
            for row in rows:
                report_id, claim_text, user_feedback, ai_label = row
                
                # Xác định label dựa trên sự khác biệt giữa user và AI
                if user_feedback == 'FAKE':
                    nli_label = 'CONTRADICTION'  # User phản bác
                elif user_feedback == 'REAL':
                    nli_label = 'ENTAILMENT'     # User xác nhận
                else:
                    nli_label = 'NEUTRAL'        # Không chắc chắn
                    
                # Thêm vào training_data
                cur.execute("""
                    INSERT INTO training_data (claim_text, evidence_text, label, source, report_id)
                    VALUES (%s, %s, %s, 'user_feedback', %s)
                """, (claim_text, claim_text, nli_label, report_id))  # evidence = claim cho đơn giản
                new_samples += 1
                
            self.conn.commit()
            print(f"📥 Đã thêm {new_samples} samples từ approved reports")
            
    def train_model(self, train_data):
        """Fine-tune CrossEncoder với dữ liệu mới"""
        
        if len(train_data) < MIN_TRAINING_SAMPLES:
            print(f"⚠️ Không đủ dữ liệu để train ({len(train_data)} < {MIN_TRAINING_SAMPLES})")
            return None
            
        print(f"\n🚀 Bắt đầu training với {len(train_data)} samples...")
        
        # Prepare InputExamples
        examples = [
            InputExample(
                texts=[d['claim'], d['evidence']],
                label=float(d['label'])
            )
            for d in train_data
        ]
        
        # Split train/val
        train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
        
        # Load base model
        print(f"   ├─ Loading base model: {BASE_MODEL_PATH}")
        model = CrossEncoder(BASE_MODEL_PATH, num_labels=3, device=self.device)
        
        # DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
        
        # Evaluator
        evaluator = None
        if len(val_examples) > 0:
            val_sentences1 = [e.texts[0] for e in val_examples]
            val_sentences2 = [e.texts[1] for e in val_examples]
            val_labels = [int(e.label) for e in val_examples]
            
        # Generate new version name
        self.new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        output_path = OUTPUT_DIR / self.new_version
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Train
        print(f"   ├─ Training for {NUM_EPOCHS} epochs...")
        model.fit(
            train_dataloader=train_dataloader,
            epochs=NUM_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            output_path=str(output_path),
            show_progress_bar=True
        )
        
        print(f"   └─ Model saved to: {output_path}")
        
        # Evaluate
        accuracy = self._evaluate_model(model, val_examples)
        
        return {
            'version': self.new_version,
            'path': str(output_path),
            'accuracy': accuracy,
            'training_samples': len(train_data)
        }
        
    def _evaluate_model(self, model, val_examples):
        """Đánh giá model trên validation set"""
        if not val_examples:
            return 0.0
            
        correct = 0
        for ex in val_examples:
            pred = model.predict([ex.texts])
            pred_label = np.argmax(pred)
            if pred_label == int(ex.label):
                correct += 1
                
        accuracy = correct / len(val_examples)
        print(f"\n📊 Validation Accuracy: {accuracy:.2%}")
        return accuracy
        
    def save_version_info(self, result):
        """Lưu thông tin version mới vào DB"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_versions (version, model_path, accuracy, training_samples)
                VALUES (%s, %s, %s, %s)
            """, (result['version'], result['path'], result['accuracy'], result['training_samples']))
            
            # Đánh dấu training_data đã dùng
            cur.execute("""
                UPDATE training_data 
                SET used_in_version = %s 
                WHERE used_in_version IS NULL
            """, (result['version'],))
            
        self.conn.commit()
        print(f"✅ Đã lưu version info: {result['version']}")
        
    def run(self):
        """Main pipeline"""
        print("=" * 60)
        print("🔄 RETRAIN PIPELINE STARTED")
        print("=" * 60)
        
        try:
            # 1. Connect DB
            self.connect_db()
            
            # 2. Chuyển approved reports thành training data
            print("\n📋 Bước 1: Xử lý approved reports...")
            self.prepare_approved_reports()
            
            # 3. Lấy training data mới
            print("\n📋 Bước 2: Lấy training data...")
            train_data = self.get_new_training_data()
            print(f"   Tìm thấy {len(train_data)} samples mới")
            
            if len(train_data) < MIN_TRAINING_SAMPLES:
                print(f"\n⏭️ Bỏ qua training: Chưa đủ {MIN_TRAINING_SAMPLES} samples")
                return False
                
            # 4. Train model
            print("\n📋 Bước 3: Training model...")
            result = self.train_model(train_data)
            
            if result:
                # 5. Save version info
                print("\n📋 Bước 4: Lưu thông tin version...")
                self.save_version_info(result)
                
                print("\n" + "=" * 60)
                print("✅ RETRAIN PIPELINE COMPLETED SUCCESSFULLY")
                print(f"   New Version: {result['version']}")
                print(f"   Accuracy: {result['accuracy']:.2%}")
                print("=" * 60)
                return True
            else:
                print("\n⚠️ Training skipped")
                return False
                
        except Exception as e:
            print(f"\n❌ Pipeline Error: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()


if __name__ == "__main__":
    pipeline = RetrainPipeline()
    success = pipeline.run()
    sys.exit(0 if success else 1)
