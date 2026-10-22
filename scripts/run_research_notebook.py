"""Execute the research notebook with the repository-local kernel, saving errors."""
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_PATH"] = str(ROOT / "runs/research_jupyter/share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.core.files import atomic_replace, remove_temporary

import nbformat
from nbclient import NotebookClient

def execute_notebook(path=None, *, client_type=NotebookClient):
    path = Path(path) if path is not None else ROOT / "output/jupyter-notebook/ihalla_vsm_training.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # A failed rerun must not retain successes from cells in an older execution.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
            cell.metadata.pop("execution", None)
    execution = {"execution_status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                 "quality_status": "unverified"}
    notebook.metadata["framegenx_execution"] = execution
    execution = notebook.metadata["framegenx_execution"]

    def save():
        fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                nbformat.write(notebook, stream)
                stream.flush()
                os.fsync(stream.fileno())
            atomic_replace(temporary, path)
        finally:
            remove_temporary(temporary)

    def started(cell, cell_index, **kwargs):
        print(f"Notebook cell {cell_index+1}/{len(notebook.cells)} running", flush=True)

    def executed(cell, cell_index, execute_reply, **kwargs):
        save()
        status = execute_reply.get("content", {}).get("status", "unknown")
        print(f"Notebook cell {cell_index+1}/{len(notebook.cells)}: {status}", flush=True)

    try:
        save()
        client_type(notebook, timeout=5500, kernel_name="framegenx-research",
                    resources={"metadata": {"path": str(ROOT)}},
                    on_cell_execute=started, on_cell_executed=executed).execute()
        execution["execution_status"] = "completed"
    except BaseException as error:
        execution.update(execution_status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error_type=type(error).__name__)
        raise
    finally:
        execution["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    print("Executed notebook: " + path.name, flush=True)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=None)
    args = parser.parse_args()
    execute_notebook(args.notebook)
