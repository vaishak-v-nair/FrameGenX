"""
FailureMemoryDB - Dynamic SRRL Failure Memory.

Logs verification failures and retrieved prompt edits to Milvus so that future
runs with similar prompts can proactively apply the fixes on Attempt 0.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from src.retrieval.milvus_client import open_local_milvus

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class FailureMemoryDB:
    def __init__(self, index_dir: str = "data/index"):
        self.index_dir = PROJECT_ROOT / index_dir
        self.milvus_path = self.index_dir / "milvus.db"
        self.collection_name = "FrameGenX_Failures"
        self._client = None
        self._dense_model = None

    def _load(self):
        if self._client is None:
            from pymilvus import MilvusClient, DataType
            from sentence_transformers import SentenceTransformer

            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._client = open_local_milvus(self.milvus_path)
            self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")

            if not self._client.has_collection(collection_name=self.collection_name):
                schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=384)
                schema.add_field(field_name="prompt", datatype=DataType.VARCHAR, max_length=65535)
                # We'll store hard_failures and edits as dynamic JSON fields by just passing them in data dict

                index_params = self._client.prepare_index_params()
                index_params.add_index(field_name="dense", index_type="FLAT", metric_type="IP")

                self._client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params
                )

    def log_failure(self, prompt: str, hard_failures: List[Dict[str, Any]], edits: List[str]):
        """Embeds the failed prompt and stores the generated edits."""
        if not edits:
            return

        self._load()

        # Use a stable hash of the prompt so identical prompts overwrite their old memory.
        # Ensure it fits within INT64 limits for Milvus primary key
        prompt_hash = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:15], 16)

        dense = self._dense_model.encode([prompt], convert_to_numpy=True).tolist()[0]

        data = {
            "id": prompt_hash,
            "dense": dense,
            "prompt": prompt,
            "hard_failures": hard_failures,
            "edits": edits
        }

        self._client.upsert(
            collection_name=self.collection_name,
            data=[data]
        )
        print(f"[memory] Logged failure for prompt: '{prompt[:40]}...' -> edits: {edits}")

    def retrieve_warnings(self, prompt: str, top_k: int = 1, threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Retrieves edits for semantically similar failed prompts."""
        try:
            self._load()
            if not self._client.has_collection(collection_name=self.collection_name):
                return []
        except Exception as e:
            print(f"[memory] Warning: Failed to load index - {e}")
            return []

        dense = self._dense_model.encode([prompt], convert_to_numpy=True).tolist()[0]

        # IP distance threshold, assuming normalized vectors.
        self._client.load_collection(self.collection_name)
        results = self._client.search(
            collection_name=self.collection_name,
            data=[dense],
            anns_field="dense",
            limit=top_k,
            output_fields=["prompt", "hard_failures", "edits"]
        )

        warnings = []
        if not results or not len(results):
            return warnings

        for hit in results[0]:
            if float(hit["distance"]) >= threshold:
                warnings.append({
                    "id": str(hit["id"]),
                    "prompt": hit["entity"].get("prompt"),
                    "hard_failures": hit["entity"].get("hard_failures", []),
                    "edits": hit["entity"].get("edits", []),
                    "score": float(hit["distance"])
                })

        return warnings

    def unload(self):
        client, self._client = self._client, None
        try:
            if client is not None:
                client.close()
        finally:
            self._dense_model = None
