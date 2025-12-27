# Fake News Detection

Comprehensive Vietnamese fake-news detection pipeline combining web scraping, Kafka-based ingestion, AI claim extraction & verification, and automated retraining with Airflow.

## Key Components
- **Crawler**: scrapes news sources and writes to a local SQLite DB (or binary Go scraper).
- **Producer**: `crawler/producer.py` reads scraped articles and pushes them to Kafka topic `raw_articles`.
- **Consumer / Processor**: `processor/consumer.py` consumes Kafka, extracts claims (PhoBERT), encodes embeddings, and stores into PostgreSQL + `pgvector`.
- **Backend API**: FastAPI in `backend/main.py` exposes `/api/v1/verify`, report endpoints and an internal `/api/internal/reload-model` for Airflow webhook.
- **Dashboard**: Streamlit admin at `dashboard/app.py` for reviewing user reports and metrics.
- **Airflow**: Orchestrates daily crawl and weekly retrain DAGs in `dags/`.
- **Model retrain pipeline**: `model/retrain_pipeline.py` (invoked by Airflow weekly DAG).

## Architecture (high level)

```
Crawler -> Kafka (raw_articles) -> Consumer (AI) -> PostgreSQL + pgvector

Users/Extension -> Backend API -> Retriever+Verifier -> Response

Airflow DAGs (daily crawl, weekly retrain) orchestrate scheduled processing and model updates.
```

## Services & URLs
- Backend API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Airflow Web: http://localhost:8080 (admin/admin)
- Airflow Scheduler: running in Docker
- Consumer (AI): processes Kafka messages (container `consumer`)
- PostgreSQL: localhost:5432 (container `db` / `my_postgres_vector`)
- PgAdmin: http://localhost:5050
- Kafka: localhost:9092
- Kafka UI: http://localhost:8888
- Zookeeper: localhost:2181

## Quickstart (Local, Docker Compose)

1. Copy env and edit if needed:

```bash
cp .env.example .env
# Edit .env for credentials (POSTGRES_USER, POSTGRES_PASSWORD, etc.)
```

2. Build & start everything (uses `start_system.sh`):

```bash
./start_system.sh --build   # builds images and starts infra + apps
./start_system.sh          # start without rebuild
```

3. Initialize application DB schema (first time only):

```bash
docker compose exec backend python init_db_full.py
```

4. Ensure Airflow connections (after UI is up): create `backend_api_connection` with Host=`backend`, Port=`8000`, Schema=`http`.

## Environment variables
Edit `.env` or `.env.example`. Important vars:
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `PGADMIN_EMAIL`, `PGADMIN_PASSWORD`
- `AIRFLOW_UID`, `_AIRFLOW_WWW_USER_USERNAME`, `_AIRFLOW_WWW_USER_PASSWORD`
- `KAFKA_BOOTSTRAP_SERVERS` (default `kafka:9093`)

## Airflow DAGs
- `dags/daily_crawl_dag.py`: runs daily crawler/processor to ingest new articles.
- `dags/weekly_retrain_dag.py`: runs weekly retrain and calls backend reload webhook.

Notes:
- Airflow requires `airflow` database; `start_system.sh` handles initialization. If Airflow reports DB errors, create DB manually in Postgres: `CREATE DATABASE airflow;` and grant privileges.

## Database schema overview
Use `init_db_full.py` to create tables:
- `articles` — raw scraped articles
- `claims` — extracted claims with embedding vector
- `users` — extension users + reputation
- `user_reports` — feedback from users
- `training_data` — approved labeled data for retraining
- `model_versions` — stored model metadata

## Model & Retraining
- Base models and tokenizers are in `/model` and `my_model*` directories.
- Weekly retrain pipeline assembles training samples from approved `user_reports`, fine-tunes a CrossEncoder, saves new version under `model/retrained_models/` and marks training data as used.

## Development & Testing
- Run backend locally:

```bash
# from repo root
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- Test verify endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/verify -H "Content-Type: application/json" -d '{"text":"Một câu cần kiểm chứng"}'
```

## Troubleshooting
- Backend import errors: ensure `backend` is a package (there is an `__init__.py`) and container has `/app` on `PYTHONPATH`.
- Airflow errors: check logs (`docker compose logs airflow-webserver`) and ensure `AIRFLOW__CORE__FERNET_KEY` is a valid Fernet key.
- Kafka issues: check `docker compose logs kafka` and `kafka-ui`.
- DB connectivity: ensure `POSTGRES_HOST` env points to `db` inside Docker, not `localhost` when inside containers.

## Helpful commands

```bash
# Show running containers
docker compose ps

# Tail backend logs
docker compose logs -f backend

# Tail consumer logs
docker compose logs -f consumer

# Recreate a single service
docker compose up -d --no-deps --build backend
```

## Contributing
- Open PRs for bug fixes and feature improvements.
- Run tests (if added) and follow code style.

---
Created/maintained by the project owner. For questions or help, open an issue or contact the maintainer.
