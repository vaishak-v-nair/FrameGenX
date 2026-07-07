import sqlite3
import json
from pathlib import Path
from typing import Dict, Any
from src.core.context import RunContext

class RunTrackerDB:
    """
    SQLite-based run tracking database.
    Logs run metadata, latencies, and success rates for benchmarking.
    """
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
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
                verification_report_json TEXT
            )
        """)
        self.conn.commit()

    def log_run(self, context: RunContext):
        """Persists the final state of a run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO runs (
                run_id, prompt, start_time, duration, is_successful, 
                repair_loops, contract_json, verification_report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context.run_id,
            context.prompt,
            context.start_time,
            context.get_duration(),
            context.is_successful,
            context.repair_loops_executed,
            json.dumps(context.contract) if context.contract else None,
            json.dumps(context.verification_report) if context.verification_report else None
        ))
        self.conn.commit()

    def get_run(self, run_id: str) -> Dict[str, Any]:
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
            "repair_loops": row[5]
        }

    def close(self):
        self.conn.close()
