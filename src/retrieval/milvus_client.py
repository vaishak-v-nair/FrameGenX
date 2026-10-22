"""Local Milvus connections owned by one retrieval or memory operation."""
from pathlib import Path


def open_local_milvus(path: Path):
    from pymilvus import MilvusClient

    # PyMilvus 3.x keeps shared channels alive after client.close(). A
    # dedicated channel actually closes with its owner. Idle 10s pings from
    # shared channels caused Milvus Lite GOAWAY and failed later memory writes.
    return MilvusClient(
        uri=str(path), dedicated=True, timeout=10.0,
        grpc_options={"grpc.keepalive_time_ms": 60000,
                      "grpc.keepalive_permit_without_calls": False},
    )
