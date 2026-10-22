"""
T1: Build benchmark_subset.json — 30 DISTINCT scenes (20 object_count +
10 color_constraints), then dry-run the IntentExtractor over every prompt
so phantom entities are caught TODAY on CPU, not overnight on GPU.

The raw dataset is 25 bases x 4 cosmetic suffix variants; naive selection
yields ~7 distinct scenes (eng review OV#2). One variant per base + authored
clean scenes fill the 30.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CURATED = [
    # --- object_count: one variant per existing base (5) ---
    {"id": "oc_cats",      "category": "object_count", "prompt": "two cats sitting on a wall"},
    {"id": "oc_apples",    "category": "object_count", "prompt": "three red apples on a table"},
    {"id": "oc_birds",     "category": "object_count", "prompt": "four birds flying in the sky"},
    {"id": "oc_books",     "category": "object_count", "prompt": "five books stacked on a desk"},
    {"id": "oc_cars",      "category": "object_count", "prompt": "two cars parked in a driveway"},
    # --- object_count: authored distinct scenes (15) ---
    {"id": "oc_dogs",      "category": "object_count", "prompt": "two dogs playing in a park"},
    {"id": "oc_bananas",   "category": "object_count", "prompt": "three bananas in a wooden bowl"},
    {"id": "oc_candles",   "category": "object_count", "prompt": "four candles on a birthday cake"},
    {"id": "oc_horses",    "category": "object_count", "prompt": "two horses standing in a field"},
    {"id": "oc_cups",      "category": "object_count", "prompt": "three cups on a kitchen counter"},
    {"id": "oc_balloons",  "category": "object_count", "prompt": "five balloons floating above a fence"},
    {"id": "oc_boats",     "category": "object_count", "prompt": "two boats on a calm lake"},
    {"id": "oc_chairs",    "category": "object_count", "prompt": "three chairs around a round table"},
    {"id": "oc_donuts",    "category": "object_count", "prompt": "four donuts on a white plate"},
    {"id": "oc_elephants", "category": "object_count", "prompt": "two elephants walking on a dirt road"},
    {"id": "oc_bottles",   "category": "object_count", "prompt": "three bottles on a shelf"},
    {"id": "oc_sunflowers","category": "object_count", "prompt": "four sunflowers in a glass vase"},
    {"id": "oc_laptops",   "category": "object_count", "prompt": "two laptops on an office desk"},
    {"id": "oc_pumpkins",  "category": "object_count", "prompt": "three pumpkins on a porch"},
    {"id": "oc_oranges",   "category": "object_count", "prompt": "five oranges in a basket"},
    # --- color_constraints: one variant per existing base (5) ---
    {"id": "cc_car_bike",  "category": "color_constraints", "prompt": "a red car parked next to a blue bicycle"},
    {"id": "cc_jacket",    "category": "color_constraints", "prompt": "a person wearing a yellow jacket and green pants"},
    {"id": "cc_cat_sofa",  "category": "color_constraints", "prompt": "a white cat sitting on a black sofa"},
    {"id": "cc_flower",    "category": "color_constraints", "prompt": "a purple flower in a brown pot"},
    {"id": "cc_laptop",    "category": "color_constraints", "prompt": "a silver laptop on a wooden desk"},
    # --- color_constraints: authored distinct scenes (5) ---
    {"id": "cc_apple",     "category": "color_constraints", "prompt": "a green apple on a white plate"},
    {"id": "cc_umbrella",  "category": "color_constraints", "prompt": "an orange umbrella leaning against a gray wall"},
    {"id": "cc_dog",       "category": "color_constraints", "prompt": "a black dog wearing a red collar"},
    {"id": "cc_mug",       "category": "color_constraints", "prompt": "a blue mug next to a silver spoon"},
    # NOTE: "a pink rose" fails here — spacy tags "rose" as a verb and the
    # entity collapses to "pink". Teapot parses cleanly and DINO detects it.
    {"id": "cc_teapot",    "category": "color_constraints", "prompt": "a pink teapot on a wooden table"},
]


def main():
    out = Path(__file__).resolve().parent.parent / "benchmark_subset.json"
    with open(out, "w") as f:
        json.dump(CURATED, f, indent=2)
    print(f"Wrote {len(CURATED)} prompts -> {out}")

    from src.intent.extractor import IntentExtractor
    ex = IntentExtractor()
    problems = 0
    for case in CURATED:
        contract = ex.extract(case["prompt"])
        ents = [(e["class_name"], len(e["attributes"]) and e["attributes"] or "")
                for e in contract["entities"]]
        counts = {}
        for e in contract["entities"]:
            counts[e["class_name"]] = counts.get(e["class_name"], 0) + 1
        flag = ""
        # Heuristic phantom check: non-countable scene words that slipped through
        suspicious = {"sky", "field", "park", "lake", "road", "driveway", "porch",
                      "shelf", "counter", "wall", "basket", "fence"}
        phantoms = [c for c in counts if c in suspicious]
        if phantoms:
            flag = f"  <-- REVIEW: scene-surface entities {phantoms}"
            problems += 1
        print(f"{case['id']:15s} {str(counts):60s}{flag}")
    print(f"\n{problems} prompts flagged for review")


if __name__ == "__main__":
    main()
