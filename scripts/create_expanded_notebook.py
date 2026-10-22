"""Fill the skill-scaffolded expanded adapter experiment with module calls."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "output/jupyter-notebook/expanded_constraint_training.ipynb"
notebook = nbf.read(path, as_version=4)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
notebook.cells = [
    md("""# Expanded constraint adapter training

**Question:** can the compatible NF4 SDXL workflow cover the expanded training
split within a bounded local run? Read the preregistered protocol before fitting.
Execution, denoising loss and image quality are separate outcomes. All adapters
remain unqualified, inactive by default, with hallucination reduction unverified.

Use the repository venv kernel. Set `HF_HOME` to your existing cached-model
location before starting Jupyter. This notebook performs no provider calls or
backbone downloads. Do not overlap fitting with tests or other model workloads.
"""),
    code("""from pathlib import Path
import os, sys, json
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/training_expanded.yaml').is_file())
sys.path.insert(0, str(root))
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from src.training.common import load_training_config, project_path
from src.training.data import validate_records, pixel_sha256, split_reservations
config = load_training_config(root / 'configs/training_expanded.yaml')
print({'model': config.model, 'resolution': config.resolution, 'steps': config.max_steps, 'quality_status': 'unverified'})"""),
    md("## 1. Validate attributed data and prior held-out reservations\nOnly positive train records supervise the generator. I-HallA and false VSR statements are held out from fitting."),
    code("""records, audit = validate_records(project_path(config.dataset))
prior, _ = split_reservations([root / 'data/training/constraint_pilot/records.jsonl'])
for row in records:
    for key in [row['image_sha256'], pixel_sha256(project_path(config.dataset).parent / row['image'])]:
        if key in prior:
            assert row['split'] == prior[key], 'Prior held-out image assignment changed'
print(audit)
assert sum(row['positive'] and row['split'] == 'train' for row in records) <= config.max_steps * config.gradient_accumulation, 'Planned steps cannot cover the full training split'"""),
    md("## 2. Check token and orientation compatibility before CUDA\nSilent caption truncation, cropping and label-changing flips are prohibited."),
    code("""from transformers import CLIPTokenizer
from PIL import Image
from src.training.model import training_model_source
from src.training.cache import checked_tokens
source = Path(training_model_source(config.model))
positives = [r for r in records if r['positive'] and r['split'] != 'test']
for folder in ['tokenizer', 'tokenizer_2']:
    tokenizer = CLIPTokenizer.from_pretrained(source / folder, local_files_only=True)
    lengths = []
    for row in positives:
        checked_tokens(tokenizer, row['caption'], row['id'])
        lengths.append(len(tokenizer(row['caption'], truncation=False, add_special_tokens=True)['input_ids']))
    print({'tokenizer': folder, 'captions_checked': len(lengths), 'max_tokens': max(lengths)})
for row in positives:
    with Image.open(project_path(config.dataset).parent / row['image']) as image:
        assert image.getexif().get(274, 1) == 1, 'EXIF orientation requires annotation review'
del tokenizer"""),
    md("## 3. Inspect the frozen protocol\n320 steps, two presentations per step, 256 px, seed 1729. Memory/deadline checks apply at explicit checkpoints, not as OS hard limits. Test images do not enter the frozen cache."),
    code("""protocol_document = root / 'docs/plans/EXPANDED_TRAINING_PROTOCOL_2026-10-17.md'
print(protocol_document.read_text(encoding='utf-8'))"""),
    md("## 4. Fit once or reuse intact evidence\nThis cell acquires the shared device lease before CUDA. A failed attempt requires explicit inspection/recovery. Identical completed reruns validate hashes and reuse their saved result."),
    code("""from src.training.study import run_study
result, reused = run_study(config, protocol_document, root / 'runs/training_studies/expanded_2026-10-17')
print({'run_id': result['run_id'], 'status': result['status'], 'reused': reused,
       'steps': result['steps_completed'], 'train_images': result['train_images'],
       'distinct_train_images_seen': result['distinct_train_images_seen'],
       'validation_diagnostic': result['validation_comparison'],
       'host_resource_samples': result['host_resource_samples'],
       'peak_reserved_gb': result['peak_reserved_gb'], 'quality_status': result['quality_status']})"""),
    md("## 5. Interpret the actual result\nFull image coverage and an intact checkpoint verify this optimization workflow. Lower denoising loss does not establish better generated counts, object/color binding or full-contract CSR. A separate paired image study with reviewed, disjoint labels must qualify the candidate. The failed scratch VSM Cards study is retained."),
    code("""assert result['promotion_status'] == 'unqualified'
assert result['quality_status'] == 'unverified'
print({'automatic_activation': False, 'hallucination_reduction': 'unverified', 'next_gate': 'disjoint paired image evaluation and human-calibrated requirement evidence'})"""),
]
notebook.metadata.kernelspec = {"display_name": "FrameGenX Research", "language": "python", "name": "framegenx-research"}
nbf.validate(notebook)
nbf.write(notebook, path)
print(f"Prepared {path.name}: {len(notebook.cells)} cells")
