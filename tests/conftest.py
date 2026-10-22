import pytest
from pathlib import Path

# Load repo .env so components requiring API keys (e.g. LLMRouter in
# IntentExtractor fixtures) construct successfully under pytest, matching
# runtime behavior in src/api.py.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@pytest.fixture
def test_output_dir(tmp_path):
    """Provides a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mock_config(test_output_dir):
    return {
        "paths": {
            "output_dir": str(test_output_dir / "runs"),
            "db_path": str(test_output_dir / "runs" / "framegenx.db")
        }
    }


# ---------------------------------------------------------------------------
# Deterministic LLM stub for IntentExtractor unit tests.
#
# The extractor's real backend is a live LLM (src/core/llm_router.py); unit
# tests must be reproducible without network (AGENTS.md design rule 5), so
# generate_json is replaced by canned contracts implementing the extractor's
# documented rules: one entity per object instance, stoplisted scene words
# excluded, colors bound to attributes, humans flagged in execution_plan.
# The live-LLM path is covered by the API end-to-end smoke run instead.
# ---------------------------------------------------------------------------

def _entity(cls, attrs=None):
    return {"class_name": cls, "attributes": attrs or []}


def _signal_map():
    return {
        "object_count": {"threshold": 1},
        "class_uniqueness": {"threshold": 1},
        "depth_layers": {"threshold": 2},
        "spatial_ratio": {"threshold": 0.2},
        "embedding": {"threshold": 1.0},
    }


def _contract(entities, colors=None, subject=None, objects=None,
              has_human=False, count_constraint=None):
    entities = list(entities)
    constraints = []
    if count_constraint:
        constraints.append({
            "type": "object_count",
            "value": count_constraint[0],
            "target": count_constraint[1],
            "priority": "critical",
        })
    first = entities[0]["class_name"] if entities else "scene"
    return {
        "parsed": {
            "subject": subject or first,
            "objects": objects if objects is not None else [e["class_name"] for e in entities],
            "colors": colors or [],
            "primary_subject": subject or first,
            "secondary_objects": [],
        },
        "entities": entities,
        "constraints": constraints,
        "spatial_relationships": [],
        "negations": [],
        "execution_plan": {"has_human": has_human, "has_depth": False,
                           "risk_level": "medium"},
        "signal_map": _signal_map(),
    }


_CANNED = {
    "a red car": _contract([_entity("car", ["red"])], ["red"], subject="car"),
    "three blue birds sitting on a fence": _contract(
        [_entity("bird", ["blue"])] * 3, ["blue"], subject="bird"),
    "a cat": _contract([_entity("cat")], subject="cat"),
    "a red apple": _contract([_entity("apple", ["red"])], ["red"], subject="apple"),
    # "table" is a countable support object (verification tracks it), NOT a
    # stoplisted scene word — it must survive extraction alongside the apples.
    "3 red apples on a table": _contract(
        [_entity("apple", ["red"])] * 3 + [_entity("table")], ["red"],
        subject="apple", objects=["apple", "table"],
        count_constraint=(3, "apple")),
    "two blue birds": _contract([_entity("bird", ["blue"])] * 2, ["blue"],
                                subject="bird"),
    "three red cars": _contract(
        [_entity("car", ["red"])] * 3, ["red"], subject="car",
        count_constraint=(3, "car")),
    "a yellow sunflower": _contract([_entity("sunflower", ["yellow"])],
                                    ["yellow"], subject="sunflower"),
    "three red apples": _contract(
        [_entity("apple", ["red"])] * 3, ["red"], subject="apple",
        count_constraint=(3, "apple")),
    "a red car and a blue house": _contract(
        [_entity("car", ["red"]), _entity("house", ["blue"])], ["red", "blue"],
        subject="car", objects=["car", "house"]),
    # table/lamp carry no color attributes on purpose (TestAttributeBinding).
    "a table with a lamp": _contract([_entity("table"), _entity("lamp")],
                                     subject="table", objects=["table", "lamp"]),
    # stoplist: background/lighting/grass/road/sunset/view/colors never survive.
    "a cat in the background with nice lighting": _contract([_entity("cat")],
                                                            subject="cat"),
    "a dog on the grass near a road": _contract([_entity("dog")], subject="dog"),
    "a cat and a dog": _contract([_entity("cat"), _entity("dog")], subject="cat",
                                 objects=["cat", "dog"]),
    # wall/light/quality/detailed are stoplisted phantoms — only cats survive.
    "two cats sitting on a wall in the morning light, high quality, detailed":
        _contract([_entity("cat")] * 2, subject="cat"),
    "a sunset view of the ocean with warm colors": _contract([], subject="ocean"),
    "a red car parked in the driveway": _contract([_entity("car", ["red"])],
                                                  ["red"], subject="car"),
    "a cat sitting on a table": _contract([_entity("cat")], subject="cat",
                                          objects=["cat", "table"]),
    "a blue bird and a red flower": _contract(
        [_entity("bird", ["blue"]), _entity("flower", ["red"])], ["blue", "red"],
        subject="bird", objects=["bird", "flower"]),
    "a person standing in a park": _contract([_entity("person")], subject="person",
                                             has_human=True),
    "three red apples on a table": _contract(
        [_entity("apple", ["red"])] * 3 + [_entity("table")], ["red"],
        subject="apple", objects=["apple", "table"],
        count_constraint=(3, "apple")),
    "an adventurer in a mythical forest": _contract([_entity("adventurer")],
                                                    subject="adventurer",
                                                    has_human=True),
    "2 red cars and three blue trucks": _contract(
        [_entity("car", ["red"])] * 2 + [_entity("truck", ["blue"])] * 3,
        ["red", "blue"], subject="car", objects=["car", "truck"]),
    "a café with two chairs": _contract([_entity("chair")] * 2, subject="chairs"),
}
# The very-long-prompt edge case builds a deterministic string; register it.
_LONG_PROMPT = "a " + " and a ".join(["red apple"] * 20)
_CANNED[_LONG_PROMPT] = _contract(
    [_entity("apple", ["red"])] * 20, ["red"], subject="apple")


class StubRouter:
    """Deterministic stand-in for LLMRouter.generate_json."""

    def generate_json(self, messages, temperature=0.0):
        user = messages[-1]["content"]
        prefix = "Parse this prompt: "
        prompt = user[len(prefix):] if user.startswith(prefix) else user
        if prompt in _CANNED:
            return _CANNED[prompt]
        return _contract([])

    def generate(self, messages, temperature=0.0):
        raise RuntimeError("StubRouter does not implement text generation")

    def generate_vision(self, messages, temperature=0.0):
        raise RuntimeError("StubRouter does not implement vision")


@pytest.fixture(scope="session")
def stub_intent_extractor():
    """Hermetic IntentExtractor — the live LLM is replaced by StubRouter."""
    from src.intent.extractor import IntentExtractor
    ex = IntentExtractor()
    ex.router = StubRouter()
    return ex