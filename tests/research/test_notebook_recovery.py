"""Completed cells survive execution failures; no real kernel/model is needed."""
import pytest

nbformat = pytest.importorskip("nbformat", reason="Optional research notebook profile")
pytest.importorskip("nbclient", reason="Optional research notebook profile")

from scripts.run_research_notebook import execute_notebook


@pytest.mark.parametrize("fault", [RuntimeError, KeyboardInterrupt])
def test_notebook_error_preserves_actual_outputs_and_original_structure(tmp_path, fault):
    path = tmp_path / "experiment.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Research"),
        nbformat.v4.new_code_cell("print('measured output')"),
        nbformat.v4.new_code_cell("raise RuntimeError('failure')"),
    ])
    nbformat.write(notebook, path)

    class FailingClient:
        def __init__(self, value, **options):
            self.notebook, self.options = value, options

        def execute(self):
            cell = self.notebook.cells[1]
            cell.execution_count = 1
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="measured output\n")]
            self.options["on_cell_executed"](cell, 1, {"content": {"status": "ok"}})
            # Outputs must already be durable before the following cell fails.
            assert nbformat.read(path, as_version=4).cells[1].execution_count == 1
            self.notebook.cells[2].execution_count = 2
            self.notebook.cells[2].outputs = [nbformat.v4.new_output(
                "error", ename=fault.__name__, evalue="injected", traceback=[])]
            raise fault("injected notebook failure")

    with pytest.raises(fault):
        execute_notebook(path, client_type=FailingClient)
    saved = nbformat.read(path, as_version=4)
    nbformat.validate(saved)
    assert [cell.source for cell in saved.cells] == [cell.source for cell in notebook.cells]
    assert saved.cells[1].outputs[0].text == "measured output\n"
    assert saved.cells[2].outputs[0].ename == fault.__name__
    assert saved.metadata.framegenx_execution.execution_status == ("interrupted" if fault is KeyboardInterrupt else "failed")
    assert not list(tmp_path.glob("*.tmp"))


def test_early_rerun_failure_cannot_retain_old_success_outputs(tmp_path):
    path = tmp_path / "experiment.ipynb"
    stale = nbformat.v4.new_code_cell("print('old success')", execution_count=9,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="OLD_SUCCESS_MUST_DISAPPEAR\n")])
    notebook = nbformat.v4.new_notebook(cells=[stale])
    nbformat.write(notebook, path)

    class StartupFailure:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self):
            raise RuntimeError("injected kernel startup failure")

    with pytest.raises(RuntimeError):
        execute_notebook(path, client_type=StartupFailure)
    saved = nbformat.read(path, as_version=4)
    assert saved.cells[0].execution_count is None and saved.cells[0].outputs == []
    assert saved.metadata.framegenx_execution.execution_status == "failed"
