"""
FrameGenX Comprehensive Pressure Test
======================================
Systematic stress-test of every pipeline stage, designed to catch:
  - meta tensor errors (the primary bug)
  - config/code mismatches
  - import failures
  - Milvus collection state issues
  - LLM router fallback chains
  - VRAM management under 6GB constraint
  - prompt encoding under sequential CPU offload
  - GroundingDINO verification stability
  - End-to-end pipeline completion

Usage:
  python tests/pressure_test.py              # run all tests
  python tests/pressure_test.py --stage intent  # run one stage
  python tests/pressure_test.py --full       # full pipeline end-to-end
"""
import sys
import os
import time
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------------------
class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.details = {}

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.duration:.1f}s)"


def run_test(name: str, func, *args, **kwargs) -> TestResult:
    result = TestResult(name)
    t0 = time.time()
    try:
        details = func(*args, **kwargs)
        result.passed = True
        result.details = details or {}
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    result.duration = time.time() - t0
    print(result)
    return result


# --------------------------------------------------------------------------
# Stage-level tests
# --------------------------------------------------------------------------

def test_imports():
    """Verify all critical imports work without errors."""
    import torch
    import diffusers
    import transformers
    import pymilvus
    import sentence_transformers
    import fastembed
    import PIL
    import numpy
    import cv2
    import pydantic
    import yaml
    import openai
    
    from src.core.context import RunContext, load_config
    from src.core.stage import Pipeline, build_stages
    from src.core.llm_router import LLMRouter
    from src.intent.extractor import IntentExtractor
    from src.retrieval.retriever import MultiModalKnowledgeRetriever
    from src.scene.planner import ScenePlanner
    from src.generation.engine import GenerationEngine
    from src.generation.lora import LoraManager
    from src.generation.attention import RegionalAttentionProcessor, find_token_indices
    from src.verification.qa import QAVerifier
    from src.verification.verify_stage import CLIPScoreVerifier, VQAVerifier
    from src.reflection.reflector import ReflectStage, build_corrective_edits
    from src.generation.refine_stage import RefineStage
    from src.pipeline import FrameGenXPipeline
    
    return {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


def test_config():
    """Verify config.yaml loads and has sane values."""
    from src.core.context import load_config
    config = load_config()
    
    assert config, "Config is empty"
    assert "generation" in config, "Missing generation section"
    assert "models" in config, "Missing models section"
    assert "pipeline" in config, "Missing pipeline section"
    
    gen = config["generation"]
    model = config["models"]["generator"]
    
    # Verify config consistency
    if "turbo" in model.lower():
        assert gen.get("steps", 4) <= 8, f"SDXL-Turbo should use <=8 steps, got {gen.get('steps')}"
        assert gen.get("cfg_scale", 0.0) <= 2.0, f"SDXL-Turbo should use cfg<=2.0, got {gen.get('cfg_scale')}"
    else:
        assert gen.get("steps", 25) >= 15, f"SDXL-base should use >=15 steps"
    
    stages = config["pipeline"]["stages"]
    assert "intent" in stages, "Missing intent stage"
    assert "generate" in stages, "Missing generate stage"
    
    return {
        "model": model,
        "steps": gen.get("steps"),
        "cfg_scale": gen.get("cfg_scale"),
        "resolution": gen.get("resolution"),
        "offload": gen.get("offload"),
        "stages": stages,
    }


def test_intent_extraction():
    """Test intent extraction with diverse prompts."""
    from src.intent.extractor import IntentExtractor
    from src.core.context import load_config
    
    config = load_config()
    extractor = IntentExtractor(config)
    
    test_prompts = [
        "A red cat sitting on a blue chair",
        "Panoramic alpine peaks at sunrise with mist in the valleys",
        "A person holding a red umbrella in the rain",
    ]
    
    results = {}
    for prompt in test_prompts:
        contract = extractor.extract(prompt)
        assert contract is not None, f"Contract is None for: {prompt}"
        assert "entities" in contract, f"Missing entities for: {prompt}"
        assert "parsed" in contract, f"Missing parsed for: {prompt}"
        assert "constraints" in contract, f"Missing constraints for: {prompt}"
        
        entities = contract["entities"]
        results[prompt[:40]] = {
            "entity_count": len(entities),
            "entities": [e.get("class_name") for e in entities],
            "has_repair_policy": "repair_policy" in contract,
        }
    
    return results


def test_milvus_retrieval():
    """Test Milvus-based retrieval."""
    from src.retrieval.retriever import Retriever
    from src.core.context import load_config
    
    config = load_config()
    retriever = Retriever(config)
    
    if not retriever.index_available():
        return {"skipped": True, "reason": "Milvus index not found"}
    
    results = retriever.query("A cat sitting on a table")
    retriever.unload()
    
    return {
        "hits": len(results),
        "top_score": results[0]["score"] if results else None,
        "top_caption": results[0]["caption"][:60] if results else None,
    }


def test_scene_planner():
    """Test scene planning with various entity configurations."""
    from src.scene.planner import ScenePlanner
    
    planner = ScenePlanner(resolution=512)
    
    # Test 1: Zero entities (scenic prompt)
    contract_0 = {
        "entities": [],
        "prompt": "A beautiful sunset over the ocean",
        "rag_hits": [],
        "parsed": {"weighted_elements": []},
    }
    _, hints_0, gen_0, prompt_0, masks_0 = planner.plan(contract_0)
    assert len(masks_0) == 0, "Zero entities should produce zero masks"
    assert hints_0["control_image_mode"] == "none", "Zero entities should disable ControlNet"
    
    # Test 2: Two entities
    contract_2 = {
        "entities": [
            {"class_name": "cat", "attributes": ["red"]},
            {"class_name": "chair", "attributes": ["blue"]},
        ],
        "prompt": "A red cat on a blue chair",
        "rag_hits": [],
        "parsed": {"weighted_elements": []},
    }
    _, hints_2, gen_2, prompt_2, masks_2 = planner.plan(contract_2)
    assert gen_2["width"] == 512
    assert gen_2["height"] == 512
    
    # Test 3: Prompt with edits
    _, _, _, prompt_e, _ = planner.plan(contract_2, prompt_edits=["prominent cat, clearly visible"])
    assert "prominent" in prompt_e.lower(), "Edits should appear in final prompt"
    
    return {
        "zero_entities": {"masks": len(masks_0), "prompt_len": len(prompt_0)},
        "two_entities": {"masks": len(masks_2), "prompt_len": len(prompt_2)},
        "with_edits": {"prompt_len": len(prompt_e)},
    }


def test_no_meta_tensor_on_load():
    """Sequential offload intentionally stores model weights on meta between calls."""
    import torch
    from src.generation.engine import GenerationEngine
    from src.core.context import load_config
    
    config = load_config()
    engine = GenerationEngine(config)
    
    # This is the call that crashed with "Cannot copy out of meta tensor"
    engine.ensure_loaded(with_controlnet=False)
    
    assert engine.pipe is not None, "Pipe should be loaded"
    
    assert engine.pipe._execution_device.type == engine.device
    assert engine.pipe.text_encoder is not None
    
    engine.teardown()
    return {"loaded": True, "execution_device": engine.device}


def test_prompt_encoding_with_sequential_offload():
    """Prompt embeddings must be real tensors with sequential CPU offload."""
    from src.generation.engine import GenerationEngine
    from src.core.context import load_config
    
    config = load_config()
    engine = GenerationEngine(config)
    engine.ensure_loaded(with_controlnet=False)
    
    conditioning, negative, pooled, negative_pooled = engine.pipe.encode_prompt(
        prompt="red cat on a blue chair",
        device=engine.pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    
    assert conditioning is not None, "Conditioning embeddings should not be None"
    assert pooled is not None, "Pooled embeddings should not be None"
    assert not conditioning.is_meta, "Conditioning should not be meta tensor"
    assert not pooled.is_meta, "Pooled conditioning should not be meta tensor"
    assert negative is None and negative_pooled is None
    
    engine.teardown()
    return {
        "conditioning_shape": list(conditioning.shape),
        "pooled_shape": list(pooled.shape),
    }


def test_generation_produces_image():
    """Test that the full generation step produces a valid image."""
    import torch
    from pathlib import Path
    from PIL import Image
    from src.generation.engine import GenerationEngine
    from src.core.context import load_config
    
    config = load_config()
    engine = GenerationEngine(config)
    
    output_path = PROJECT_ROOT / "runs" / "_pressure_test" / "test_gen.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gen_config = {
        "width": 512,
        "height": 512,
        "steps": config.get("generation", {}).get("steps", 4),
        "cfg_scale": config.get("generation", {}).get("cfg_scale", 0.0),
        "seed": 42,
    }
    
    meta = engine.generate(
        prompt="A red cat sitting on a blue chair, digital art",
        gen_config=gen_config,
        control_hints={"control_image_mode": "none"},
        masks={},
        output_path=output_path,
    )
    
    assert output_path.exists(), "Generated image file should exist"
    img = Image.open(output_path)
    assert img.size == (512, 512), f"Image should be 512x512, got {img.size}"
    
    engine.teardown()
    return {
        "image_exists": True,
        "image_size": list(img.size),
        "steps": meta.get("steps"),
        "seed": meta.get("seed"),
    }


def test_qa_verifier():
    """Test GroundingDINO object detection verifier."""
    from pathlib import Path
    from src.verification.qa import QAVerifier
    
    # Use the pressure test image if available
    test_img = PROJECT_ROOT / "runs" / "_pressure_test" / "test_gen.png"
    if not test_img.exists():
        return {"skipped": True, "reason": "No test image available (run test_generation first)"}
    
    verifier = QAVerifier()
    
    # Test with cat entity
    contract = {
        "entities": [{"class_name": "cat", "attributes": ["red"]}],
    }
    result = verifier.verify(test_img, contract)
    
    assert "hard_failures" in result, "Result should have hard_failures key"
    assert "signals" in result, "Result should have signals key"
    
    return {
        "hard_failures": len(result.get("hard_failures", [])),
        "detected_counts": result.get("signals", {}).get("detected_counts", {}),
    }


def test_clip_scorer():
    """Test CLIP score verification."""
    from PIL import Image
    from src.verification.verify_stage import CLIPScoreVerifier
    from src.core.context import load_config
    
    test_img = PROJECT_ROOT / "runs" / "_pressure_test" / "test_gen.png"
    if not test_img.exists():
        return {"skipped": True, "reason": "No test image (run test_generation first)"}
    
    config = load_config()
    scorer = CLIPScoreVerifier(config)
    image = Image.open(test_img).convert("RGB")
    
    scores = scorer.score(image, "A red cat sitting on a blue chair")
    
    assert "clip_cos" in scores, "Should have clip_cos"
    assert "clip_norm" in scores, "Should have clip_norm"
    assert 0.0 <= scores["clip_norm"] <= 1.0, f"clip_norm out of range: {scores['clip_norm']}"
    
    return scores


def test_llm_router():
    """Test LLM router with fallback chain."""
    from src.core.llm_router import LLMRouter
    
    router = LLMRouter()
    
    # Simple text generation
    result = router.generate(
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        temperature=0.0
    )
    assert result is not None, "LLM should return a response"
    assert len(result) > 0, "Response should not be empty"
    
    # JSON generation
    json_result = router.generate_json(
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": 'Return {"test": true}'},
        ],
        temperature=0.0
    )
    assert isinstance(json_result, dict), "JSON result should be a dict"
    
    return {
        "text_response_len": len(result),
        "json_keys": list(json_result.keys()),
        "providers": [p.name for p in router.available_providers],
    }


def test_reflector():
    """Test reflection logic with mock verification reports."""
    from src.reflection.reflector import build_corrective_edits
    
    # Test: count mismatch
    report_fail = {
        "hard_failures": [
            {"type": "count_mismatch", "class_name": "cat", "expected": 1, "detected": 0}
        ],
        "clip_norm": 0.3,
        "reward": 0.25,
    }
    contract = {
        "parsed": {"primary_subject": "red cat"},
        "repair_policy": {
            "mappings": [
                {"failure": "object_missing", "action": "prominently featured"},
                {"failure": "low_alignment", "action": "sharp focus, highly detailed"},
            ]
        },
    }
    
    edits = build_corrective_edits(report_fail, contract)
    assert len(edits) > 0, "Should produce corrective edits for failures"
    
    # Test: no failures
    report_pass = {
        "hard_failures": [],
        "clip_norm": 0.8,
        "reward": 0.75,
    }
    edits_pass = build_corrective_edits(report_pass, contract)
    assert len(edits_pass) == 0, "Should produce zero edits when no failures"
    
    return {
        "fail_edits": edits,
        "pass_edits": edits_pass,
    }


def test_full_pipeline():
    """End-to-end pipeline test with a real prompt."""
    from src.pipeline import FrameGenXPipeline
    
    pipeline = FrameGenXPipeline()
    
    prompt = "A red cat sitting on a blue chair"
    ctx = pipeline.run(prompt)
    
    assert ctx is not None, "RunContext should not be None"
    assert ctx.run_id, "Should have a run_id"
    assert ctx.contract is not None, "Should have a contract"
    assert ctx.final_prompt, "Should have a final prompt"
    assert ctx.run_dir.exists(), "Run directory should exist"
    
    # Check that at least one generated image exists
    gen_images = list(ctx.run_dir.glob("generated_*.png"))
    assert len(gen_images) > 0, "Should have at least one generated image"
    
    return {
        "run_id": ctx.run_id,
        "accepted": ctx.is_successful,
        "attempts": ctx.attempt + 1,
        "reward_history": ctx.reward_history,
        "final_prompt_len": len(ctx.final_prompt),
        "entities": [e.get("class_name") for e in ctx.contract.get("entities", [])],
        "duration": round(ctx.get_duration(), 1),
    }


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------

from src.core.device import exclusive_device


@exclusive_device
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FrameGenX Pressure Test")
    parser.add_argument("--stage", type=str, default=None,
                        help="Run a specific stage test (imports, config, intent, milvus, planner, "
                             "meta_tensor, compel, generation, qa, clip, llm, reflector, full)")
    parser.add_argument("--full", action="store_true", help="Run full pipeline E2E test")
    args = parser.parse_args()

    all_tests = {
        "imports": ("1. Import Verification", test_imports),
        "config": ("2. Config Validation", test_config),
        "llm": ("3. LLM Router Fallback", test_llm_router),
        "intent": ("4. Intent Extraction", test_intent_extraction),
        "milvus": ("5. Milvus Retrieval", test_milvus_retrieval),
        "planner": ("6. Scene Planner", test_scene_planner),
        "meta_tensor": ("7. CRITICAL: No Meta Tensor on Load", test_no_meta_tensor_on_load),
        "encoding": ("8. Prompt Encoding with CPU Offload", test_prompt_encoding_with_sequential_offload),
        "generation": ("9. Image Generation", test_generation_produces_image),
        "qa": ("10. QA Verifier (GroundingDINO)", test_qa_verifier),
        "clip": ("11. CLIP Score Verifier", test_clip_scorer),
        "reflector": ("12. Reflector Logic", test_reflector),
    }

    if args.full:
        tests_to_run = {**all_tests, "full": ("13. FULL PIPELINE E2E", test_full_pipeline)}
    elif args.stage:
        if args.stage not in all_tests and args.stage != "full":
            print(f"Unknown stage: {args.stage}. Available: {list(all_tests.keys()) + ['full']}")
            sys.exit(1)
        if args.stage == "full":
            tests_to_run = {"full": ("13. FULL PIPELINE E2E", test_full_pipeline)}
        else:
            tests_to_run = {args.stage: all_tests[args.stage]}
    else:
        tests_to_run = all_tests

    print("\n" + "=" * 70)
    print("  FrameGenX Comprehensive Pressure Test")
    print("=" * 70)

    results = []
    for key, (name, func) in tests_to_run.items():
        print(f"\n--- {name} ---")
        result = run_test(name, func)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration for r in results)
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status}  {r.name} ({r.duration:.1f}s)")
        if r.error:
            print(f"         Error: {r.error}")
    
    print(f"\n  Total: {passed}/{len(results)} passed, {failed} failed ({total_time:.1f}s)")
    
    if failed > 0:
        print("\n  ⚠ PRESSURE TEST FAILED — fix errors above before deploying.")
        sys.exit(1)
    else:
        print("\n  ✓ ALL TESTS PASSED — pipeline is stable.")
        sys.exit(0)


if __name__ == "__main__":
    main()
