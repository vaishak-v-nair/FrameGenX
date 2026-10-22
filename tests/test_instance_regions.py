from copy import deepcopy
from types import SimpleNamespace
import re

import pytest
import torch

from src.generation.regions import (compile_regions, instance_attention,
                                     InstanceAttentionProcessor, region_weights)


def contract():
    return {"prompt": "One red car to the left of one blue bus on a gray road.",
            "entities": [{"id": "car", "class_name": "car", "attributes": ["red"]},
                         {"id": "bus", "class_name": "bus", "attributes": ["blue"]},
                         {"id": "road", "class_name": "road", "attributes": ["gray"]}],
            "spatial_relationships": [{"subject": "car", "object": "bus", "predicate": "left_of"}]}


def test_region_prompts_keep_nouns_and_colors_separate_without_mutating_contract():
    given = contract(); before = deepcopy(given)
    plan = compile_regions(given)
    assert given == before
    assert len(plan["regions"]) == 2
    assert "red car" in plan["regions"][0]["prompt"] and not re.search(r"\bblue\b", plan["regions"][0]["prompt"])
    assert "blue bus" in plan["regions"][1]["prompt"] and not re.search(r"\bred\b", plan["regions"][1]["prompt"])
    assert plan["background_prompt"] == "gray road."
    assert plan["regions"][0]["box"][2] < plan["regions"][1]["box"][0]


def test_repeated_classes_get_independent_instances():
    plan = compile_regions({"prompt": "three red apples in a row", "entities": [
        {"id": str(i), "class_name": "apple", "attributes": ["red"]} for i in range(3)]})
    assert len({tuple(r["box"]) for r in plan["regions"]}) == 3
    assert all("one single red apple" in r["prompt"] for r in plan["regions"])


@pytest.mark.parametrize("predicate", ["on", "behind", "between", "unknown"])
def test_unsupported_foreground_relations_do_not_become_disjoint_layouts(predicate):
    given = contract(); given["spatial_relationships"][0]["predicate"] = predicate
    assert compile_regions(given)["enabled"] is False


def test_direction_and_contradiction_are_handled():
    given = contract(); given["spatial_relationships"][0]["predicate"] = "right_of"
    assert compile_regions(given)["regions"][0]["class_name"] == "bus"
    given["spatial_relationships"].append({"subject": "car", "object": "bus", "predicate": "left_of"})
    assert compile_regions(given)["enabled"] is False


def test_uncounted_foreground_and_out_of_budget_counts_are_ineligible():
    given = contract(); given["entities"][0]["uncounted"] = True
    assert compile_regions(given)["enabled"] is False
    assert compile_regions({"entities": [{"class_name": "apple"}] * 7})["enabled"] is False


@pytest.mark.parametrize("placement", [{"relative_position": "right"},
    {"bounding_box": {"xmin": 300, "ymin": 10, "xmax": 500, "ymax": 400}}])
def test_explicit_contract_placement_cannot_be_silently_replaced(placement):
    given = contract()
    given["entities"][0].update(placement)
    assert compile_regions(given)["enabled"] is False


def test_duplicate_entity_identity_is_not_silently_rebound():
    given = contract()
    given["entities"][1]["id"] = "car"
    assert compile_regions(given)["enabled"] is False


@pytest.mark.parametrize("size", [8, 16, 64])
def test_region_masks_cover_every_pixel_once_at_all_attention_scales(size):
    weights = region_weights(compile_regions(contract()), size, size, "cpu", torch.float32)
    assert torch.allclose(weights.sum(0), torch.ones(size, size))
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert all(w.sum() > 0 for w in weights)


def test_attention_routes_conditional_context_and_preserves_unconditional():
    plan = compile_regions(contract())
    plan["regions"][0]["box"] = [0, 0, .5, 1]
    plan["regions"][1]["box"] = [.5, 0, 1, 1]
    def original(attn, hidden, context, *args, **kwargs):
        return context.mean((1, 2))[:, None, None].expand_as(hidden)
    proc = InstanceAttentionProcessor(original, plan, [torch.full((1, 3, 4), v) for v in (0., 2., 5.)], True)
    hidden = torch.zeros(2, 16, 4)
    context = torch.full((2, 3, 4), 9.)
    output = proc(None, hidden, context)
    assert torch.equal(output[0], torch.full_like(output[0], 9.))
    assert torch.equal(output[1].reshape(4, 4, 4)[:, :2], torch.full((4, 2, 4), 2.))
    assert torch.equal(output[1].reshape(4, 4, 4)[:, 2:], torch.full((4, 2, 4), 5.))


def test_exact_processors_restored_even_when_generation_fails():
    first, second = object(), object()
    class Unet:
        attn_processors = {"x.attn1.processor": first, "x.attn2.processor": second}
        def set_attn_processor(self, processors): self.attn_processors = processors
    unet = Unet(); plan = compile_regions(contract())
    with pytest.raises(RuntimeError):
        with instance_attention(unet, plan, [torch.zeros(1, 3, 4)] * 3, False):
            assert unet.attn_processors["x.attn1.processor"] is first
            raise RuntimeError("synthetic sampling failure")
    assert unet.attn_processors == {"x.attn1.processor": first, "x.attn2.processor": second}


def test_native_diffusers_attention_retains_unconditional_normalization_and_residual():
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    torch.manual_seed(1)
    attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4,
                     residual_connection=True, rescale_output_factor=2)
    original = AttnProcessor2_0()
    hidden, context = torch.randn(2, 16, 8), torch.randn(2, 3, 8)
    baseline = original(attn, hidden, context)
    proc = InstanceAttentionProcessor(original, compile_regions(contract()),
                                      [torch.randn(1, 3, 8) for _ in range(3)], True)
    actual = proc(attn, hidden, context)
    assert torch.allclose(baseline[0], actual[0], atol=1e-6)
