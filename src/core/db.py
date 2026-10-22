import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.core.context import RunContext


class RunTrackerDB:
    """
    SQLite-based run tracking database.
    Logs run metadata, latencies, and success rates for benchmarking.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                prompt TEXT,
                start_time REAL,
                duration REAL,
                is_successful BOOLEAN,
                repair_loops INTEGER,
                contract_json TEXT,
                verification_report_json TEXT,
                config_name TEXT DEFAULT 'interactive',
                metrics_json TEXT,
                is_benchmark BOOLEAN DEFAULT 0,
                benchmark_seed INTEGER,
                cache_key TEXT
            )
        """)
        # Migration for pre-existing DBs: CREATE IF NOT EXISTS won't add new
        # columns, and the widened INSERT would fail with "no such column".
        cursor.execute("PRAGMA table_info(runs)")
        existing = {row[1] for row in cursor.fetchall()}
        if "config_name" not in existing:
            cursor.execute("ALTER TABLE runs ADD COLUMN config_name TEXT DEFAULT 'full'")
        if "metrics_json" not in existing:
            cursor.execute("ALTER TABLE runs ADD COLUMN metrics_json TEXT")
        if "is_benchmark" not in existing:
            cursor.execute("ALTER TABLE runs ADD COLUMN is_benchmark BOOLEAN DEFAULT 0")
        if "benchmark_seed" not in existing:
            cursor.execute("ALTER TABLE runs ADD COLUMN benchmark_seed INTEGER")
        if "cache_key" not in existing:
            cursor.execute("ALTER TABLE runs ADD COLUMN cache_key TEXT")
        self.conn.commit()

    def log_run(self, context: RunContext):
        """Persists the final state of a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, prompt, start_time, duration, is_successful,
                repair_loops, contract_json, verification_report_json,
                config_name, metrics_json, is_benchmark, benchmark_seed, cache_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                context.run_id,
                context.prompt,
                context.start_time,
                context.get_duration(),
                context.is_successful,
                context.repair_loops_executed,
                json.dumps(context.contract) if context.contract else None,
                json.dumps(context.verification_report)
                if context.verification_report
                else None,
                context.config_name,
                json.dumps(context.metrics()),
                context.is_benchmark,
                context.benchmark_seed,
                context.cache_key,
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "prompt": row[1],
            "duration": row[3],
            "is_successful": bool(row[4]),
            "repair_loops": row[5],
        }

    def get_run_by_prompt(self, prompt: str, cache_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Only interactive runs may be reused in the studio or CLI.
        cursor = self.conn.cursor()
        query = (
            "SELECT * FROM runs WHERE prompt = ? AND is_successful = 1 "
            "AND config_name = 'interactive'"
        )
        params = [prompt]
        if cache_key is not None:
            query += " AND cache_key = ?"
            params.append(cache_key)
        query += " ORDER BY start_time DESC LIMIT 1"
        cursor.execute(query, params)
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "prompt": row[1],
            "duration": row[3],
            "is_successful": bool(row[4]),
            "repair_loops": row[5],
            "verification_report_json": row[7],
        }

    def has_run(self, prompt: str, config_name: str, seed: Optional[int] = None,
                experiment_key: Optional[str] = None) -> bool:
        """Completed benchmark rows count for resume; infrastructure errors do not."""
        cursor = self.conn.cursor()
        query = ("SELECT verification_report_json FROM runs WHERE prompt = ? "
                 "AND config_name = ? AND is_benchmark = 1 AND benchmark_seed IS ?")
        params = [prompt, config_name, seed]
        if experiment_key is not None:
            query += " AND cache_key = ?"
            params.append(experiment_key)
        cursor.execute(query, params)
        return any(not self._is_run_error(row[0]) for row in cursor.fetchall())

    @staticmethod
    def _is_run_error(report_json: Optional[str]) -> bool:
        if not report_json:
            return False
        try:
            return bool(json.loads(report_json).get("error"))
        except (TypeError, ValueError, AttributeError):
            return False

    def latest_benchmark_experiment(self) -> Optional[str]:
        """Newest fingerprinted suite; old unlabeled rows remain archival."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT cache_key FROM runs WHERE is_benchmark = 1 "
            "AND cache_key LIKE 'benchmark-v2:%' "
            "AND config_name IN ('baseline', 'rag', 'rag_lora', 'full') "
            "ORDER BY start_time DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def all_benchmark_rows(self, experiment_key: Optional[str] = None):
        """Rows from one reproducible suite; never aggregate different revisions."""
        experiment_key = experiment_key or self.latest_benchmark_experiment()
        if not experiment_key:
            return []
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT config_name, prompt, is_successful, repair_loops, metrics_json, "
            "verification_report_json "
            "FROM runs WHERE metrics_json IS NOT NULL AND is_benchmark = 1 "
            "AND config_name IN ('baseline', 'rag', 'rag_lora', 'full') "
            "AND cache_key = ?",
            (experiment_key,),
        )
        return [row[:5] for row in cursor.fetchall()
                if not self._is_run_error(row[5])]

    def close(self):
        self.conn.close()
