# Fake News Detection — Project Report

> Concise technical report documenting architecture, models, data, and operational workflow for the Vietnamese Fake News Detection system.

---

## 🚀 Project Summary
- **Purpose:** Detect misinformation in Vietnamese news via a hybrid pipeline (claim extraction + vector retrieval + Cross-encoder verification).
- **Key features:** Automated ingestion (crawler → Kafka), claim extraction (PhoBERT), vector KB (pgvector), Cross-Encoder NLI verifier, browser extension for user feedback, Airflow-driven retraining loop.

---

## 🔧 Tech Stack
- Language & frameworks: Python 3.x, FastAPI, Streamlit
- ML infra: PyTorch, Hugging Face Transformers, SentenceTransformers
- DB & vector search: PostgreSQL + pgvector
- Orchestration: Apache Airflow (DAGs in `dags/`)
- Messaging: Kafka (topic `raw_articles`)
- Packaging & runtime: Docker & docker-compose, Uvicorn
- Vietnamese NLP helper: `underthesea` (sentence tokenization)

---

## 📁 Important Files & Components
- Crawling & ingestion: `crawler/producer.py`, `dags/daily_crawl_dag.py`
- Claim extraction & KB build: `processor/rebuild_knowledge_base.py`, `scripts/build_knowledge_base.py`
- Retriever: `sentence_transformers` bi-encoder (`bkai-foundation-models/vietnamese-bi-encoder`)
- Claim detector: `model/train_claim_detector.py` → saved in `claim_detector_model/`
- Verifier: CrossEncoder models (e.g., `my_model_v7/`) loaded by `backend/verifier.py` and `test/verifier.py`
- Retrain pipeline: `model/retrain_pipeline.py` (called by `dags/weekly_retrain_dag.py`)
- Backend API: `backend/main.py` (endpoints: `/api/v1/verify`, `/api/v1/report`, `/api/internal/reload-model`)
- UI & Extension: `dashboard/app.py`, `extension/popup.js`
- Synthetic / training data: `scripts/generate_nli_data.py`, `smart_train_data.csv`, `data/nli_train.json`

---

## 🧠 Models & Training Details
### Claim Detector
- Base: `vinai/phobert-base-v2` (PhoBERT)
- Task: binary sequence classification (Claim / Non-Claim)
- Training script: `model/train_claim_detector.py`
  - Tokenization: max_length=128
  - Training: 2 epochs, LR=2e-5, per-device batch=16
  - Output: `claim_detector_model/` (tokenizer + model)

### Retriever (Bi-Encoder)
- Model: `SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')`
- Purpose: encode claims & KB sentences; embeddings stored in `claims.embedding` (pgvector)
- Querying: SQL with `<=>` operator; acceptance threshold used in code: distance < 0.5

### Verifier (Cross-Encoder / NLI)
- Approach: CrossEncoder fine-tuned for 3 classes: REFUTES(0), SUPPORTS(1), NEI(2)
- Quick training script: `scripts/train_nli_model.py`
  - MODEL_NAME = `vinai/phobert-base-v2`
  - BATCH_SIZE = 4, EPOCHS = 2, MAX_SAMPLES = 15000, MAX_SEQ_LENGTH = 256
  - Output path example: `model/my_model_v3_fast`
- Production retrain: `model/retrain_pipeline.py`
  - Base model path: env var `BASE_MODEL_PATH` (default `my_model_v7`)
  - TRAIN_BATCH_SIZE = 16, NUM_EPOCHS = 3, LEARNING_RATE = 2e-5, WARMUP_STEPS = 100
  - MIN_TRAINING_SAMPLES = 50
  - Output saved to `model/retrained_models/<version>`

### Synthetic & Human-Labeled Data
- `scripts/generate_nli_data.py` produces synthetic NLI training triples: SUPPORTS (same sentence), REFUTES (hard negatives via date/number/entity/negation swaps), NEUTRAL (random sentence)
- `smart_train_data.csv` contains many synthetic pairs (sentence1, sentence2, label)
- Feedback-based samples are generated from `user_reports` and approved by admins, then written to `training_data` table by `model/retrain_pipeline.py`.

---

## 🔁 End-to-End Workflow
1. **Ingestion (Daily)**
   - Crawler scrapes sources; `dags/daily_crawl_dag.py` runs crawler and `processor/rebuild_knowledge_base.py`.
   - Rebuilder: sentence-split → candidate filter → claim detector → bi-encoder encode → insert into `claims` with `system_label='REAL'`.

2. **Verification (On-demand)**
   - User/extension sends page text to `/api/v1/verify`.
   - `AdvancedFactChecker.verify`:
     - Extract claims (sentence tokenize)
     - Encode with retriever and query nearest (pgvector)
     - If close enough, run CrossEncoder on (claim, evidence)
     - Softmax logits → label & confidence
     - Decision engine applies rules (strong refute → FAKE, majority support → REAL, else NEUTRAL)

3. **Feedback Loop & Retraining (Weekly)**
   - Extension allows users to report individual claims (`/api/v1/report`).
   - Admin reviews in Streamlit dashboard (`dashboard/app.py`) and approves.
   - `model/retrain_pipeline.py` collects approved reports → converts to NLI training examples → fine-tunes CrossEncoder → saves new version → records in `model_versions` and marks training_data used.
   - Airflow DAG `dags/weekly_retrain_dag.py` runs retrain and calls backend internal reload endpoint to load new model.

---

## 🗄️ Database Overview (Core Tables)
- `articles`: raw scraped articles
- `claims`: extracted claims with `embedding` (pgvector), `system_label`, `verified` and metadata
- `users`: extension users and `reputation_score`
- `user_reports`: feedback rows (linked to `claims`) with admin `status`
- `training_data`: NLI training rows with `used_in_version` flag
- `model_versions`: records of retrained model versions and metrics

---

## 🧪 Testing & Logs
- Basic tests & checks: `test_api_config.py`, `test_api_endpoints.py`, `test_ner.py`
- Airflow logs are in `logs/` (e.g., `logs/scheduler/latest/weekly_retrain_dag.py.log`)
- Model training logs written by `sentence-transformers` / Transformers during training; validation accuracy stored in retrain pipeline result and DB

---

## 🔧 Deployment & Run (Quick commands)
- Copy env & start system (with docker/docker-compose):
```bash
cp .env.example .env
./start_system.sh --build   # build & start stack
./start_system.sh           # start without rebuild
```
- Run backend locally for development:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Rebuild knowledge base manually:
```bash
python processor/rebuild_knowledge_base.py
```
- Start retrain pipeline locally (manual run):
```bash
python model/retrain_pipeline.py
```

---

## ✅ Recommendations & Next Steps
- Add more robust evaluation (per-class precision/recall/F1, confusion matrix) and store metrics in DB.
- Add unit/integration tests for retrain pipeline and model reload flow.
- Add model validation & sanity checks in `retrain_pipeline.py` before releasing a new version.
- Monitor model drift and data distribution changes; consider automatic alerts if KB distribution shifts.
- For scale, consider FAISS or a production HNSW-based pgvector config to support large KBs.

---

## 📦 Artifacts & Where to Find Them
- Claim detector: `claim_detector_model/`
- Verifier models: `my_model_v7/`, `my_model_v6/` (and `model/retrained_models/` after retrain)
- Synthetic train sets: `smart_train_data.csv`, `data/nli_train.json`
- Retrain results: `model/retrained_models/<version>` and DB table `model_versions`

---

## Contact & Credits
- Maintained by project owner (repo root contact info / README license).

---

If you'd like, I can:
- Add a compact architecture diagram (Mermaid) into this `REPORT.md` ✅
- Generate a one-page executive summary (PDF/MD) focused on non-technical stakeholders ✅
- Add CI checks for training & post-training validation into the repo ✅

Tell me which of the above you want next.