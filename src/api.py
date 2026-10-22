import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from typing import Optional, Dict, Any  # noqa: E402
import uvicorn  # noqa: E402
from pathlib import Path  # noqa: E402
import json  # noqa: E402
import base64  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

from src.pipeline import FrameGenXPipeline  # noqa: E402
from src.intent.models import ContractV2  # noqa: E402
from src.research.api import router as research_router  # noqa: E402
from src.core.device import device_lease, DeviceBusyError  # noqa: E402

app = FastAPI(title="FrameGenX API", version="0.2.0")
app.include_router(research_router)
logger = logging.getLogger(__name__)
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRAMEGENX_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
runs_dir = PROJECT_ROOT / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)

RUN_IMAGE_NAME = re.compile(r"(?:generated_\d+|global_mask_\d+|step_\d+_\d+)\.png")


@app.get("/runs/{run_id}/{filename}")
def get_run_image(run_id: str, filename: str):
    try:
        if str(uuid.UUID(run_id)) != run_id or not RUN_IMAGE_NAME.fullmatch(filename):
            raise ValueError("Invalid run image path")
    except ValueError:
        raise HTTPException(status_code=404, detail="Run image not found.")
    path = runs_dir / run_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run image not found.")
    return FileResponse(path, media_type="image/png")

# Serve the built frontend from the API itself — one process, no node
# dev server eating RAM on a 16GB laptop. Rebuild with: cd frontend && npm run build
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/app", StaticFiles(directory=str(_dist), html=True), name="app")
    if (_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="assets")

pipeline = FrameGenXPipeline()

# ---------------------------------------------------------------------------
# Glass-box job API: one generation at a time (single 6GB GPU), progress
# events streamed via polling, per-stage artifacts exposed as they land.
# ---------------------------------------------------------------------------

JOBS: Dict[str, Dict[str, Any]] = {}
GPU_BUSY = threading.Lock()


class JobRequest(BaseModel):
    prompt: str = Field(max_length=2000)
    force: bool = False  # skip the cache and generate fresh


class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, max_length=2000)
    contract: Optional[ContractV2] = None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_artifacts(run_id: str) -> Dict[str, Any]:
    """Everything the pipeline wrote for this run, stage by stage.
    Safe to call mid-run: files that don't exist yet are simply absent."""
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        return {"run_id": run_id, "attempts": []}

    bundle: Dict[str, Any] = {
        "run_id": run_id,
        "contract": _read_json(run_dir / "contract_v2.json"),
        "rag_hits": _read_json(run_dir / "rag_hits.json"),
        "reward_history": _read_json(run_dir / "reward_history.json"),
        "attempts": [],
    }
    for n in range(0, 8):
        has_any = any(
            (run_dir / f"{stem}_{n}{ext}").exists()
            for stem, ext in [
                ("scene_plan", ".json"), ("generated", ".png"),
                ("verification", ".json"), ("decision", ".json"),
            ]
        )
        if not has_any:
            break
        final_prompt = None
        fp = run_dir / f"final_prompt_{n}.txt"
        if fp.exists():
            try:
                final_prompt = fp.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        attempt = {
            "attempt": n,
            "scene_plan": _read_json(run_dir / f"scene_plan_{n}.json"),
            "final_prompt": final_prompt,
            "image_url": f"/runs/{run_id}/generated_{n}.png"
            if (run_dir / f"generated_{n}.png").exists() else None,
            "mask_url": f"/runs/{run_id}/global_mask_{n}.png"
            if (run_dir / f"global_mask_{n}.png").exists() else None,
            "intermediate_steps": [
                f"/runs/{run_id}/step_{n}_{i}.png"
                for i in range(10)
                if (run_dir / f"step_{n}_{i}.png").exists()
            ],
            "verification": _read_json(run_dir / f"verification_{n}.json"),
            "decision": _read_json(run_dir / f"decision_{n}.json"),
        }
        bundle["attempts"].append(attempt)
    return bundle


def _run_job(job: Dict[str, Any], admission=None):
    def progress(evt: Dict[str, Any]):
        job["events"].append(evt)
        if evt["event"] == "run_started":
            job["run_id"] = evt["run_id"]
        elif evt["event"] == "stage_started":
            job["stage"] = evt["stage"]
            job["attempt"] = evt.get("attempt", 0)
        elif evt["event"] == "decision":
            job["attempt"] = evt.get("attempt", 0)

    try:
        with device_lease():
            if admission is not None:
                admission.set()
            ctx = pipeline.run(job["prompt"], progress_cb=progress)
            job["run_id"] = ctx.run_id
            job["accepted"] = bool(ctx.is_successful)
            job["status"] = "done"
    except DeviceBusyError as error:
        job["status"] = "error"
        job["error_code"] = error.code
        job["error"] = str(error)
    except Exception as e:
        job["status"] = "error"
        job["error_code"] = "generation_failed"
        job["error"] = "Generation failed. Check the local model and runtime configuration before retrying."
        # Provider exceptions may contain credential-bearing request URLs.
        logger.error("Generation worker failed (%s)", type(e).__name__)
    finally:
        job["finished_at"] = time.time()
        GPU_BUSY.release()
        if admission is not None:
            admission.set()


@app.post("/jobs")
def create_job(req: JobRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    # Cache: a previously accepted run for this exact prompt returns instantly.
    if not req.force:
        cached = pipeline.db.get_run_by_prompt(prompt, cache_key=pipeline.cache_key)
        if cached and (runs_dir / cached["run_id"] / f'generated_{cached["repair_loops"]}.png').is_file():
            job_id = uuid.uuid4().hex[:12]
            JOBS[job_id] = {
                "job_id": job_id, "prompt": prompt, "status": "done",
                "cached": True, "run_id": cached["run_id"],
                "accepted": cached["is_successful"], "stage": None,
                "attempt": cached.get("repair_loops", 0), "events": [],
                "error": None, "started_at": time.time(),
                "finished_at": time.time(),
            }
            return {"job_id": job_id, "cached": True}

    if not GPU_BUSY.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A generation is already running. One at a time on a 6GB GPU "
                   "— wait for the current run to finish (or check the "
                   "benchmark isn't running).",
        )

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id, "prompt": prompt, "status": "running",
        "cached": False, "run_id": None, "accepted": None, "stage": None,
        "attempt": 0, "events": [], "error": None,
        "started_at": time.time(), "finished_at": None,
    }
    JOBS[job_id] = job
    admission = threading.Event()
    try:
        threading.Thread(target=_run_job, args=(job, admission), daemon=True).start()
    except Exception:
        JOBS.pop(job_id, None)
        GPU_BUSY.release()
        raise HTTPException(status_code=503, detail="Could not start the local worker.") from None
    # The worker acquires the OS lease in its own thread. No cross-thread lock
    # ownership transfer, and a busy CLI/training process is visible immediately.
    # On a slow OS the job remains inspectable by its returned handle.
    admission.wait(timeout=1.0)
    if job.get("error_code") == "device_busy":
        JOBS.pop(job_id, None)
        raise HTTPException(status_code=409, detail={"code": "device_busy", "message": job["error"]})
    return {"job_id": job_id, "cached": False}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    out = dict(job)
    out["elapsed"] = round(
        (out["finished_at"] or time.time()) - out["started_at"], 1
    )
    return out


@app.get("/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if not job.get("run_id"):
        return {"run_id": None, "attempts": []}
    return _collect_artifacts(job["run_id"])


@app.get("/artifacts/{run_id}")
def get_artifacts(run_id: str):
    # Direct access for showcase runs (e.g. the SRRL evidence run).
    try:
        if str(uuid.UUID(run_id)) != run_id or not (runs_dir / run_id).is_dir():
            raise ValueError("Unknown run id")
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown run id.")
    return _collect_artifacts(run_id)


# ---------------------------------------------------------------------------
# Dashboard endpoints: aggregated metrics and run history.
# ---------------------------------------------------------------------------

@app.get("/stats")
def get_stats():
    """Aggregated benchmark metrics for the ablation dashboard."""
    from collections import defaultdict

    experiment_key = pipeline.db.latest_benchmark_experiment()
    rows = pipeline.db.all_benchmark_rows(experiment_key=experiment_key)
    by_config = defaultdict(list)
    for config_name, prompt, ok, loops, metrics_json in rows:
        m = json.loads(metrics_json) if metrics_json else {}
        m["ok"] = bool(ok)
        m["loops"] = loops
        m["prompt"] = prompt
        by_config[config_name].append(m)

    # Load category map from benchmark dataset
    categories = {}
    for ds_file in ["benchmark_subset.json", "benchmark_dataset.json"]:
        ds_path = PROJECT_ROOT / ds_file
        if ds_path.exists():
            try:
                with open(ds_path, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        categories[item["prompt"]] = item.get("category", "unknown")
            except Exception:
                pass

    config_stats = {}
    for cfg, runs in by_config.items():
        n = len(runs)
        if n == 0:
            continue
        hard_pass_rate = sum(r["ok"] for r in runs) / n

        # Per-category breakdown
        cat_data = defaultdict(lambda: {"passed": 0, "total": 0})
        for r in runs:
            cat = categories.get(r.get("prompt", ""), "unknown")
            cat_data[cat]["total"] += 1
            if r["ok"]:
                cat_data[cat]["passed"] += 1

        rewards = [r.get("reward") for r in runs if r.get("reward") is not None]
        clips = [r.get("clip_norm") for r in runs if r.get("clip_norm") is not None]
        vqas = [r.get("vqa_score") for r in runs if r.get("vqa_score") is not None]
        attempts = [r.get("attempts") for r in runs if r.get("attempts") is not None]
        latencies = [r.get("latency_s") for r in runs if r.get("latency_s") is not None]

        def safe_mean(xs):
            return sum(xs) / len(xs) if xs else None

        config_stats[cfg] = {
            "n": n,
            "hard_pass_rate": round(hard_pass_rate, 3),
            "clip_norm": round(safe_mean(clips), 4) if clips else None,
            "vqa_score": round(safe_mean(vqas), 4) if vqas else None,
            "reward": round(safe_mean(rewards), 4) if rewards else None,
            "avg_attempts": round(safe_mean(attempts), 2) if attempts else None,
            "avg_latency": round(safe_mean(latencies), 1) if latencies else None,
            "rewards": [round(r, 4) for r in rewards],
            "by_category": {
                cat: {
                    "hard_pass_rate": round(d["passed"] / d["total"], 3) if d["total"] > 0 else 0,
                    "n": d["total"],
                }
                for cat, d in cat_data.items()
            },
        }

    return {
        "metric_scope": "count_and_spatial_hard_checks",
        "experiment_key": experiment_key,
        "provenance_scope": "latest_fingerprinted_suite",
        "configs": config_stats,
        "total_runs": sum(s["n"] for s in config_stats.values()),
    }


@app.get("/api/runs")
def list_runs(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    config: str = None,
    accepted_only: bool = False,
):
    """Paginated run history list."""
    cursor = pipeline.db.conn.cursor()

    where_clauses = ["metrics_json IS NOT NULL"]
    params = []
    if config:
        where_clauses.append("config_name = ?")
        params.append(config)
    if accepted_only:
        where_clauses.append("is_successful = 1")

    where = " AND ".join(where_clauses)
    offset = (page - 1) * per_page

    # Count total
    cursor.execute(f"SELECT COUNT(*) FROM runs WHERE {where}", params)
    total = cursor.fetchone()[0]

    # Fetch page
    cursor.execute(
        f"SELECT run_id, prompt, is_successful, repair_loops, config_name, "
        f"duration, metrics_json, verification_report_json FROM runs WHERE {where} "
        f"ORDER BY start_time DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    )
    rows = cursor.fetchall()

    runs_list = []
    for row in rows:
        m = json.loads(row[6]) if row[6] else {}
        verification = json.loads(row[7]) if row[7] else {}
        status = "error" if verification.get("error") else (
            "accepted" if row[2] else "rejected"
        )
        runs_list.append({
            "run_id": row[0],
            "prompt": row[1],
            "accepted": bool(row[2]),
            "status": status,
            "repair_loops": row[3],
            "config_name": row[4],
            "duration": round(row[5], 1) if row[5] else None,
            "reward": m.get("reward"),
            "clip_norm": m.get("clip_norm"),
            "vqa_score": m.get("vqa_score"),
            "has_artifacts": (runs_dir / row[0]).exists() if row[0] else False,
        })

    return {
        "runs": runs_list,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


# ---------------------------------------------------------------------------
# Legacy synchronous endpoint (kept for /docs playground and old clients).
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "FrameGenX API",
        "version": "0.2.0",
        "docs_url": "/docs"
    }

class GenerateResponse(BaseModel):
    run_id: str
    success: bool
    verification_report: Optional[Dict[str, Any]] = None
    image_base64: Optional[str] = None

@app.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest):
    if request.contract is not None:
        raise HTTPException(status_code=400, detail="Contract submission is not supported by this endpoint; submit a prompt to /jobs.")
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    prompt = request.prompt.strip()

    # 1. Check Local SQLite Cache (Instant Return)
    cached_run = pipeline.db.get_run_by_prompt(prompt, cache_key=pipeline.cache_key)
    if cached_run:
        print(f"Local Cache Hit for prompt: {prompt}")
        run_id = cached_run["run_id"]
        run_dir = runs_dir / run_id

        # Load the cached image as base64
        image_path = run_dir / f'generated_{cached_run["repair_loops"]}.png'

        img_b64 = None
        if image_path.is_file():
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

        if img_b64:
            verification_report = {}
            if cached_run.get("verification_report_json"):
                verification_report = json.loads(cached_run["verification_report_json"])

            return GenerateResponse(
                run_id=run_id,
                success=cached_run.get("is_successful", True),
                verification_report=verification_report,
                image_base64=img_b64
            )

    if not GPU_BUSY.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A generation is already running.")
    try:
        # 2. Generate Image via SDXL-Turbo Local Pipeline
        with device_lease():
            ctx = pipeline.run(prompt)

        # The final report certifies ctx.attempt. Return that same image.
        img_b64 = None
        image_path = ctx.run_dir / f"generated_{ctx.attempt}.png"
        if image_path.exists():
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

        response_data = GenerateResponse(
            run_id=ctx.run_id,
            success=ctx.is_successful,
            verification_report=ctx.verification_report,
            image_base64=img_b64
        )

        return response_data

    except DeviceBusyError as error:
        raise HTTPException(status_code=409, detail={"code": error.code, "message": str(error)}) from None
    except Exception as error:
        logger.error("Generation failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Generation failed; check the local model and runtime configuration.") from None
    finally:
        GPU_BUSY.release()

if __name__ == "__main__":
    # reload=False: the reloader can restart the server mid-generation.
    uvicorn.run("src.api:app", host=os.getenv("FRAMEGENX_HOST", "127.0.0.1"),
                port=8000, reload=False)
