# DeepCue — Deployment Guide

This covers two scenarios: running everything on your own machine for development,
and deploying to a production server. Model training (Kaggle) is documented
separately in [kaggle_scripts/README.md](kaggle_scripts/README.md).

---

## 1. Local Development Setup

DeepCue has **four independent processes** that all need to run simultaneously.
There is no single "start" command — this is intentional, since each maps to a
separate concern (infra, WS/HTTP server, async task worker, static frontend).

### 1.1 Prerequisites

- Python 3.12+ (3.14 used in development)
- Docker Desktop (for Redis + local MongoDB)
- A MongoDB Atlas account (production data store) — or use the local MongoDB
  container from `docker-compose.yml` for dev-only testing
- Trained ONNX model files (see [kaggle_scripts/README.md](kaggle_scripts/README.md))
  placed under `models/video/`, `models/audio/`, `models/text/`, `models/fusion/`

### 1.2 Environment variables

Copy `.env.example` to `.env` in the project root and fill in:

```env
DJANGO_SECRET_KEY=<any long random string>
DJANGO_DEBUG=True
DJANGO_SETTINGS_MODULE=deepcue_backend.settings.local

MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/deepcue?appName=Cluster0
MONGODB_DB_NAME=deepcue

REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
CHANNELS_REDIS_URL=redis://localhost:6379/2

VIDEO_MODEL_PATH=models/video/efficientnet_lstm_quant.onnx
AUDIO_MODEL_PATH=models/audio/wav2vec2_classifier_quant.onnx
TEXT_MODEL_PATH=models/text/xlm_roberta_sentiment_quant.onnx
FUSION_MODEL_PATH=models/fusion/cross_modal_transformer_quant.onnx

EMAIL_HOST_USER=<gmail address>
EMAIL_HOST_PASSWORD=<gmail App Password, not your real password>
```

**MongoDB Atlas note:** whitelist your current IP under Network Access (or
`0.0.0.0/0` for dev only — never in production).

**Gmail App Password:** Google Account → Security → 2-Step Verification → App
Passwords. A regular account password will not work for SMTP.

### 1.3 Install dependencies

```bash
cd backend
pip install -r ../requirements.txt
python manage.py migrate
```

`migrate` sets up Django's own SQLite database (admin/sessions) — separate
from MongoDB, which stores the actual interview data.

### 1.4 Start infrastructure (Redis + optional local MongoDB)

```bash
docker compose up -d
```

This starts Redis (`localhost:6379`) and, if you're not using Atlas locally,
a MongoDB container + Mongo Express UI at `http://localhost:8081`.

### 1.5 Start the backend (Django Channels via Daphne)

```bash
cd backend
python -m daphne -b 0.0.0.0 -p 8000 deepcue_backend.asgi:application
```

Use `python -m daphne`, not the bare `daphne` command — on Windows, pip-installed
console scripts often aren't on `PATH` even though the package is installed.

There is no homepage at `http://localhost:8000/` — this process only serves
the WebSocket endpoint (`ws://localhost:8000/ws/interview/<uuid>/`) and the
`/api/...` HTTP endpoints. A 404 on `/` is expected.

### 1.6 Start the Celery worker

```bash
cd backend
python -m celery -A deepcue_backend worker -l info --pool=solo
```

**`--pool=solo` is required on Windows** — Celery's default `prefork` pool
uses `os.fork()`, which doesn't exist on Windows and will fail silently or
raise `ValueError`. `--pool=solo` runs tasks single-threaded in the same
process; fine for development. On Linux production servers, omit this flag
to get the default multi-process pool.

### 1.7 Serve the frontend

```bash
cd frontend
python serve.py
```

Then open **`http://localhost:5500`** (the landing page).

**Do not use `python -m http.server` directly** — on Windows, Python's
`http.server` resolves `.js` files to `text/plain` via a broken Windows
registry MIME mapping. Browsers refuse to execute `<script type="module">`
files served with the wrong MIME type, which silently breaks every button
on the page with no visible error besides a console warning. `serve.py`
patches `mimetypes` before starting the server specifically to avoid this.

If you still see "Failed to load module script... MIME type text/plain"
after using `serve.py`, check for a **stale duplicate process** still bound
to port 5500 (`netstat -ano | grep :5500` on Windows) — kill it and restart;
Windows allows multiple processes to bind the same port, and the OS may
route requests to the old one.

### 1.8 Verify the full stack

Open `http://localhost:5500`, fill in the landing form, click Continue, then
"Start Interview." You should see a 5-second countdown, then a camera/mic
permission prompt. If the permission prompt never appears, open DevTools →
Console first — it almost always means a JS module failed to load (see 1.7),
not a WebSocket or camera-permissions issue.

---

## 2. Running Tests

```bash
cd backend
python -m pytest -q
```

Tests use `deepcue_backend.settings.test`, an in-memory Channels layer, and
`CELERY_TASK_ALWAYS_EAGER=True` — no real Redis, MongoDB, or ONNX models are
needed to run the test suite.

---

## 3. Production Deployment

### 3.1 Process layout

Same four processes as local dev, but each hardened:

| Process | Local dev | Production |
|---|---|---|
| Infra | `docker compose up` (Redis + local Mongo) | Managed Redis (e.g. ElastiCache) + MongoDB Atlas (already cloud-hosted) |
| Backend | `daphne` directly | `daphne` behind Nginx (reverse proxy + TLS termination), managed by systemd/supervisor |
| Worker | `celery worker --pool=solo` | `celery worker` (default prefork pool, no `--pool=solo`), run via systemd, autoscale with `--autoscale` |
| Frontend | `serve.py` | Static files served by Nginx directly, or a CDN — no Python server needed in production |

### 3.2 Settings module

Set `DJANGO_SETTINGS_MODULE=deepcue_backend.settings.production`, which enables:
- `DEBUG=False`
- `SECURE_SSL_REDIRECT`, HSTS, secure cookies
- `ALLOWED_HOSTS` / `CORS_ALLOWED_ORIGINS` read strictly from env vars (no wildcard)

### 3.3 Logging

Production settings automatically switch to **structured JSON logs** (one
JSON object per line to stdout) instead of the human-readable format used in
dev — controlled by the `DJANGO_DEBUG` env var, not the Django `DEBUG`
setting itself, so it's correct before settings overrides apply. Point your
log aggregator (CloudWatch, Datadog, etc.) at stdout of the Daphne/Celery
processes.

### 3.4 Secrets

Never commit `.env`. In production, inject `DJANGO_SECRET_KEY`,
`MONGODB_URI`, `EMAIL_HOST_PASSWORD`, etc. via your platform's secrets
manager (AWS Secrets Manager, Doppler, etc.) rather than a checked-in file.

### 3.5 Model files

The four `*_quant.onnx` files (and any `.onnx.data` companion files — see
note below) must be present on the server's filesystem at the paths defined
by `VIDEO_MODEL_PATH` / `AUDIO_MODEL_PATH` / `TEXT_MODEL_PATH` /
`FUSION_MODEL_PATH`. These are binary artifacts, not code — deploy them via
your artifact pipeline (e.g. bundled into the Docker image, or pulled from
S3 on container start), not via git.

**Important:** some exported ONNX models split weights into a separate
`<name>.onnx.data` file alongside the `.onnx` graph file. Both must be
deployed together in the same directory, or `onnxruntime.InferenceSession`
will fail to load the model.

### 3.6 Rate limiting & validation

The WebSocket consumer already enforces per-connection rate limits
(`apps/sessions_app/rate_limit.py`) and Pydantic schema validation
(`apps/sessions_app/validation.py`) on every inbound message — no additional
reverse-proxy-level rate limiting is required for the WS endpoint itself,
though you should still rate-limit at the Nginx/load-balancer level against
connection-flooding (many new WebSocket connections per second from one IP).

### 3.7 Health check

`GET /api/health/` returns service status for Django, Redis, and MongoDB —
point your load balancer's health check here.
