"""Fill the skill-scaffolded experiment with tested-module orchestration cells."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "output/jupyter-notebook/ihalla_vsm_training.ipynb"
notebook = nbf.read(path, as_version=4)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
notebook.cells = [
    md("""# I-HallA evaluation and VSM Cards ML/DL feasibility

Objective: run a reproducible classical-ML diagnostic and an equal-budget learned-variance diffusion comparison on licensed VSM Cards data. I-HallA stays a held-out factual-QA evaluation source until image rights/assets/provider permissions are available.

**This is CPU research, not SDXL fine-tuning or a qualified hallucination-reduction result.** General full-contract CSR remains unverified. See `docs/plans/RESEARCH_TRAINING_EXECUTION_PLAN.md` for the plan completed before coding and the separate execution tasks.

Sources: [I-HallA](https://github.com/kaist-cvml/I-HallA-v1.0), [VSM](https://github.com/bhosalems/VSM), [VSM dataset CC BY 4.0](https://huggingface.co/datasets/mbhosale/VSM), [VSM paper](https://arxiv.org/abs/2606.00377)."""),
    md("## 1. Setup and reproducibility\nUse the FrameGenX CPU Research kernel. No provider key, model download or CUDA operation is requested. Asset download is a separate audited task; the cells fail if required local data are missing or changed."),
    code("""from pathlib import Path
import os, sys, json
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
checkout = next((p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / 'configs/research_cards.json').is_file()), None)
if checkout is None:
    raise RuntimeError('Open this notebook from the FrameGenX checkout.')
os.chdir(checkout)
if str(checkout) not in sys.path:
    sys.path.insert(0, str(checkout))
from src.research.config import load_config
from src.research.artifacts import sha256
config = load_config()
assert config.device == 'cpu'
from threadpoolctl import threadpool_limits
cpu_threads = threadpool_limits(limits=config.threads)
print({'python': sys.version.split()[0], 'device': config.device,
       'protocol_sha256': sha256(checkout / 'configs/research_cards.json')})"""),
    md("## 2. Frozen hypotheses and evaluation\nBaseline rho=0 versus candidate rho=0.1, same initialization, training data/noise/steps and learned-variance sampler. Use final fixed-step checkpoints. Thirty-two paired sampling seeds are feasibility evidence only. The objective follows the paper's inverse-square-root schedule; the pinned repository's constant-rho variant differs. Do not tune the test threshold or choose a favorite checkpoint after seeing outputs."),
    code("""print({k: getattr(config, k) for k in ['steps','batch_size','diffusion_steps','seed',
    'candidate_rho','vsm_schedule','ncc_threshold','ambiguity_margin','sample_seeds']})"""),
    md("## 3. Validate data and disjoint splits\nVerify both pinned archives, safely import a bounded subset, require all 40 RGB templates, and group complete card-grid identities before fitting. Nonimage archive scripts are ignored and never run. Unknown source labels are retained in the manifest and excluded from positive diffusion training."),
    code("""from src.research.cards import prepare_data, load_data
manifest_path = prepare_data(config)
manifest, matcher = load_data(manifest_path, config)
print({'records': len(manifest['records']), 'template_count': len(manifest['templates']),
       'verified_split_counts': manifest['verified_split_counts'],
       'manifest_sha256': sha256(manifest_path), 'license': manifest['license']})"""),
    md("## 4. Inspect I-HallA availability\nPublished benchmark facts/questions remain held out. Code licensing does not establish rights to every externally hosted image. No external VQA or fine-tuning runs from this cell; unavailable scores remain null."),
    code("""from src.research.ihalla import inspect_ihalla
ihalla = inspect_ihalla(checkout / 'reports/research_sources/ihalla_vsm_2026-10-16/ihalla')
print(ihalla)"""),
    md("## 5. Train classical ML and both DL arms, then sample\nReusable modules perform fitting, checkpoints, CPU budgets, MLflow tracking and evaluation. This cell can take several minutes. It reuses only an intact completed run with the exact config/data/code identity; interruption leaves failed/incomplete evidence and a new attempt restarts both arms. It does not pretend model-only checkpoints resume optimizer/RNG state."),
    code("""from src.research.experiment import run_experiment
report_path = run_experiment('configs/research_cards.json', reuse_completed=True)
report = json.loads(report_path.read_text(encoding='utf-8'))
print({'run_id': report['id'], 'execution_status': report['execution_status'],
       'quality_status': report['quality_status'], 'report': str(report_path.relative_to(checkout))})"""),
    md("## 6. Actual results and uncertainty\nML metrics concern named synthetic corruptions. DL rates concern color-aware Cards template validity. Neither establishes generic contract satisfaction. Every planned output counts; blank/noise/failure is not filtered away. The paired interval covers sampling-seed variation for this single training pair, not training-seed or full-domain uncertainty."),
    code("""summary = {
    'classical_test': report['classical_ml']['metrics']['test'],
    'domain_evaluation': report['evaluation'],
    'resources': report['resources'],
    'sdxl_activation': report['sdxl_integration']['activation']
}
print(json.dumps(summary, indent=2))"""),
    md("## 7. Fixed sample inspection\nInspect the first two preregistered seeds from each arm; no best-image selection. These pixels remain unchanged in the evidence bundle."),
    code("""from IPython.display import display
from PIL import Image
for arm in ['baseline', 'candidate']:
    print(arm)
    for seed in config.sample_seeds[:2]:
        with Image.open(report_path.parent / arm / 'samples' / f'seed_{seed}.png') as image:
            display(image.copy())"""),
    md("""## 8. Interpretation and next steps

- A finished training lifecycle is not image-quality qualification. Keep all models experimental.
- The RGB six-channel Cards checkpoint is incompatible with the current four-latent-channel SDXL LoRA path. Do not rename or load it as an adapter.
- The SDXL learned-variance head/sampler/export probe and a disjoint human-labeled full-contract study remain required before generic activation.
- I-HallA fine-tuning remains gated on image rights, reviewed factual labels and a separate training partition; this notebook never trains on its held-out evaluation questions/images.
- FrameGenX's read-only research API/UI consumes the versioned report. Advanced tests check data/model/notebook/report and existing application behavior; browser/hardware/hosted evidence is labeled separately.
- ChatGPT Dots is pending identification; this notebook does not claim an unavailable service ran.
"""),
]
notebook.metadata.kernelspec = {"display_name": "FrameGenX CPU Research", "language": "python", "name": "framegenx-research"}
notebook.metadata.language_info = {"name": "python"}
nbf.validate(notebook)
nbf.write(notebook, path)
print(path.relative_to(ROOT))
