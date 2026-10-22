# FrameGenX image-generation research and implementation review

Date: 2026-10-13; implementation and verification updated 2026-10-14.
Scope: current FrameGenX checkout and primary research or
official implementation sources. The target machine is an RTX 3050 Laptop GPU
with 6GB VRAM and 16GB RAM. This report records engineering decisions, not a
new claim of improved constraint satisfaction.

## What the evidence says

The main risk is verification coverage. FrameGenX can operationally pass object
counts and supported box relations while leaving color, other attributes,
negation, uncounted presence, and arbitrary extracted constraints unverified.
The saved red-cat/blue-chair run illustrates this: its archived report accepted
an `on` relation with the reason `unknown predicate 'on': skipped`, and its
color VQA was unavailable. The spatial verifier was fixed in the previous QA
review; the new evidence ledger also treats that archived skipped result as
unverified. A benchmark pass rate based on only count/spatial checks is not
full prompt-level CSR.

## Research and project decision

| Primary source | Relevant result | FrameGenX decision |
| --- | --- | --- |
| [GenEval](https://arxiv.org/abs/2310.11513) and [GenEval 2](https://arxiv.org/abs/2512.16853) | Object-centric count, position, and color checks are more diagnostic than a holistic score. GenEval 2 documents benchmark drift and reports up to 17.7 percentage points of disagreement with human judgment on recent models. | Keep detector evidence per atom and audit the judge against human labels before calling its result CSR. Add a newer compositional holdout; do not treat the current 30-prompt hard-pass rate as full satisfaction. |
| [TIFA](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_TIFA_Accurate_and_Interpretable_Text-to-Image_Faithfulness_Evaluation_with_Question_Answering_ICCV_2023_paper.pdf) and [Davidsonian Scene Graph](https://arxiv.org/abs/2310.18235) | Atomic questions improve interpretability; DSG explicitly handles duplicate, missing, and dependent questions. | Make the extracted contract an atom inventory. Count every unsupported requirement as unverified. Add dependency-aware questions only after extraction recall is measured. |
| [VQAScore](https://arxiv.org/abs/2404.01291) and [GenAI-Bench](https://arxiv.org/abs/2406.13743) | VQA-based whole-prompt scoring can outperform CLIP alignment on compositional prompts and can rank multiple candidate images, but fine details remain a limitation. | Retain CLIP and VQA as diagnostics. Evaluate small best-of-N candidate selection only after atomic checks, GPU cost, and human alignment are measured. |
| [NegT2IBench](https://arxiv.org/abs/2610.03084) and [NegBench](https://arxiv.org/abs/2501.09425) | Negated requirements need separate polarity scoring; vision-language models can misread negation. The recently published NegT2IBench reports detector-based polarity evaluation with human comparison. | Keep negatives as unverified today. Add positive/negative paired prompts and calibrate absence detection before it becomes an acceptance gate. A detector miss alone is not proof of absence. |
| [PSGEval](https://openaccess.thecvf.com/content/ICCV2025/html/Deng_Leveraging_Panoptic_Scene_Graph_for_Evaluating_Fine-Grained_Text-to-Image_Generation_ICCV_2025_paper.html) | Scene-graph and instance matching address object/attribute/relation assignment and expose limitations of simple detector metrics. | Replace first-box spatial decisions with explicit instance identities and matching. Until then, multi-instance spatial passes are marked unverified in the ledger. |
| [EPIC](https://arxiv.org/abs/2605.11722) | A fixed predicate program drives verification and routes failed local predicates to editing and global failures to resampling. Its reported gains and reduced model/MLLM calls are on its own setup, not FrameGenX. | Preserve one immutable contract across repair attempts. Prototype targeted edit/resample routing with a fixed call budget and reverify every edited image; benchmark against the current loop before adopting it. |
| [BoxDiff](https://arxiv.org/abs/2307.10816) and [GLIGEN](https://arxiv.org/abs/2301.07093) | Explicit box conditioning can improve spatial control; BoxDiff is training-free, while GLIGEN adds trained grounding layers. | Test BoxDiff-like attention guidance as an isolated 6GB experiment after the spatial judge is calibrated. Do not add another default model load without measured VRAM, latency, and CSR benefit. |
| [Diffusers memory guide](https://huggingface.co/docs/diffusers/main/en/optimization/memory) and [SDXL-Turbo guide](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/sdxl_turbo) | Sequential CPU offload saves VRAM but can be slow; SDXL-Turbo is designed for 512px, guidance 0, and 1–4 denoising steps. | Keep the locally measured sequential-offload configuration and 512px/4-step generator. Benchmark any offload change on the laptop rather than importing a paper's high-memory setting. |

## Changes applied in this checkout

1. `src/verification/contract_evidence.py` records each extracted count,
   attribute, relation, negation, and other constraint as passed, failed, or
   unverified. It records VQA answers as observations, not proof. Empty
   extraction and old skipped relations cannot become full-contract passes.
   `src/verification/verify_stage.py` persists the ledger per attempt; the
   studio shows the coverage and requirements needing review.
2. Vision QA parsing now abstains on malformed or incomplete answers. It no
   longer treats `not red` or arbitrary prose as a matching color, and a
   partial batch cannot produce an apparently complete reward score. A
   zero-question contract has no VQA score rather than a synthetic 1.0.
3. GroundingDINO count processing now suppresses near-identical boxes of the
   same class at a conservative 0.90 IoU threshold. Raw and retained counts
   are saved for diagnosis. This threshold still needs labeled calibration.
4. Default post-acceptance refinement is off, avoiding an extra SDXL pass and
   an unverified derivative. Explicit refinement remains available and its
   metadata labels `refined_final.png` as unverified; the API displays the
   checked draft.
5. Benchmark runs now write a manifest fingerprinting backend code/config,
   runner, selected cases, and installed package versions. Resume and the
   dashboard use one fingerprinted suite. Older unlabeled rows remain
   archival. The manifest records configured model references but cannot yet
   prove the exact downloaded weight revisions.
6. Planning now copies entities into `scene_plan` instead of mutating the
   extracted contract with generated layout boxes. Generation consumes that
   plan; verification retains the original requirements. Explicit support
   verbs such as `sitting on` are preserved when the extractor weakens an
   already bound relation to `above`; corrections are recorded in the contract.
7. Offline Diffusers loading resolves the cached marker file to its snapshot
   directory using [the Hub's local-file API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).
   A partially cached repository containing the required fp16 pipeline files
   now loads without downloading unrelated repository files. Missing required
   components still fail. The resolved generator source is recorded in metadata.
8. Provider fallback now tries each eligible provider once per request and
   respects cooldowns. SDK retries are disabled so a configured 30-second
   timeout does not multiply through hidden attempts; the
   [official SDK documents its default retries](https://github.com/openai/openai-python#retries).
   A live run reached Gemini after Ollama failed and Nvidia timed out, and
   obtained diagnostic answers instead of retrying the unavailable local route.
9. The final-step callback had decoded the SDXL VAE a second time and saved
   an all-black preview while the final image was valid. Previews now reuse
   the final pipeline decode, eliminating that extra decode. The UI labels
   them saved previews. Default engine settings consistently use SDXL-Turbo,
   sequential offload, and opt-in ControlNet.
10. Retrieval now reads `spatial_relationships`, recognizes `on`, excludes
    uncounted entities from layout counts, and binds supporting-surface boxes
    to subject/object roles in either entity order. It previously selected
    a side by side layout for the red-cat/blue-chair support relation.
11. PyMilvus 3.x shared clients retained idle channels after `close()`, causing
    `too_many_pings` GOAWAY and a failed later repair-memory write. Local clients
    now use dedicated channels, no keepalive pings without calls, and a
    60-second keepalive interval. Client ownership matches the
    [PyMilvus connection manager](https://github.com/milvus-io/pymilvus/blob/master/pymilvus/client/connection_manager.py);
    the keepalive policy follows [gRPC's guidance](https://grpc.io/docs/guides/keepalive/).
    Retrieval, failure memory, and index construction use the same factory.
12. The synchronous `/generate` response now returns the final checked image
    with its own verification report. A higher diagnostic reward on an earlier
    rejected draft can no longer pair that image with a later draft's report.

## Verification evidence on 2026-10-14

| Check | Recorded result and scope |
| --- | --- |
| Backend unit/API/regression suite | **182 passed in 108.85s**, excluding the separate live-model pressure suite. [Log](qa_pytest_2026-10-14.log), [exit status](qa_pytest_2026-10-14.exit). Earlier silent/interrupted attempts are not counted as passes. |
| Frontend | Lint and production build passed; all 4 saved-run state tests passed. These checks do not establish rendered browser or accessibility behavior. |
| Bounded integrated run | [Run 624ef7cb](../runs/624ef7cb-2f4d-4ae8-8895-2c8e2f8201ce/verification_0.json) used the configured checkpoint ID offline, seed 42, one attempt, no refinement. It completed in **159.3s** and was rejected by the preserved `on` predicate. Ledger: 2 operational count passes, 1 spatial failure, 2 unverified colors. Gemini answered `orange` for the cat and `blue` for the chair; VQA score 0.5 is diagnostic. |
| Preview/memory smoke after the final fixes | [Run 4461f5db](../runs/4461f5db-2cf8-44cd-8c90-5ccfbc4c7947/smoke_result.json) passed actual GPU generation, nonblank preview equality, contract preservation, and synthetic repair-memory write/reopen/read. Generation **32.19s**, total **41.05s**; process peak allocated VRAM **2.777 GB**, reserved **2.915 GB**. This measures this process, not total GPU/display usage. [Log](qa_research_smoke_2026-10-14.log). |

The integrated run exposed two separate visual limits. The cat's fur is
orange despite the red requirement. It is visibly supported by the chair,
but the enclosing-chair box includes the backrest, so the box-top contact
heuristic rejects the relation. Preserving `on` fixed extraction semantics;
it did not calibrate this judge. Color binding and support-surface geometry
remain priorities for the labeled judge evaluation below. The final smoke
also produced orange fur and makes no contract-satisfaction claim.

The isolated smoke is reproducible with:

```powershell
$env:HF_HOME='E:\hf_cache'
venv\Scripts\python.exe scripts\research_smoke.py --source-run 624ef7cb-2f4d-4ae8-8895-2c8e2f8201ce
```

The script uses offline model loading, a saved contract, and a new synthetic
memory database under its new run folder. It does not write to the production
retrieval index or invoke external providers. The earlier browser-policy block
and unavailable Docker daemon still limit the broad QA assessment. No fresh
multi-seed ablation or calibrated full-CSR result is claimed.

## Next experiments before stronger claims

1. **Contract recall:** label prompt atoms independently for a stratified
   sample across count, binding, relations, negation, and uncounted presence.
   Compare the original prompt with the extracted contract. A perfect visual
   verifier cannot recover an omitted atom.
2. **Judge calibration:** annotate generated images independently and measure
   per-category false-pass rate, false-reject rate, coverage/abstention, and
   agreement with human labels. Include detector duplicates, overlapping
   instances, color swaps, unsupported relations, and absent objects. Freeze
   thresholds before testing the holdout.
3. **Instance-aware binding:** give contract entities stable IDs and match
   detected boxes to them before evaluating crop-level attributes and
   relations. Keep box-area depth proxies and unknown predicates unverified.
4. **Repair budget:** compare the current loop with fixed-program targeted
   editing, global resampling, and a small candidate-ranking budget under
   paired seeds. Record prompt-level satisfaction, GPU peak, RAM peak,
   latency, provider calls, and model invocations. Reverify every derivative.
5. **Reproducibility:** capture resolved model and LoRA checkpoint commit IDs
   or file hashes, then run a fresh multi-seed benchmark. Report full
   prompt-level CSR only where every prompt atom has independently validated
   evidence; continue to label the existing count/spatial rate separately.

The paper results above are research findings in their own evaluation setups.
No paper-reported improvement is presented as a measured FrameGenX gain.
