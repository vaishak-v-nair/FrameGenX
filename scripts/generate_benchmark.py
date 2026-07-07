import json

# Define the categories and templates
categories = {
    "object_count": [
        "two cats sitting on a wall",
        "three red apples on a table",
        "four birds flying in the sky",
        "five books stacked on a desk",
        "two cars parked in a driveway"
    ],
    "spatial_relations": [
        "a dog sitting under a table",
        "a cup placed next to a laptop",
        "a person standing behind a counter",
        "a bird perched on top of a streetlamp",
        "a book inside a glass cabinet"
    ],
    "text_rendering": [
        "a neon sign that says CAFE",
        "a protest banner with the text FREEDOM",
        "a t-shirt with the word HELLO printed on it",
        "a billboard advertising COLA",
        "a chalkboard with WELCOME written on it"
    ],
    "multi_object_scenes": [
        "a desk with a laptop, a coffee mug, and a plant",
        "a kitchen counter with a blender, a knife, and a cutting board",
        "a living room with a sofa, a television, and a rug",
        "a park scene with a bench, a trash can, and a bicycle",
        "a dining table with a plate, a fork, and a glass of water"
    ],
    "color_constraints": [
        "a red car parked next to a blue bicycle",
        "a person wearing a yellow jacket and green pants",
        "a white cat sitting on a black sofa",
        "a purple flower in a brown pot",
        "a silver laptop on a wooden desk"
    ]
}

# Generate 20 prompts per category by expanding the variations
prompts = []
for category, base_prompts in categories.items():
    # To hit 20 per category, we will do 4 variations per base prompt
    for i in range(len(base_prompts)):
        base = base_prompts[i]
        prompts.append({"id": f"{category}_{i*4+1}", "category": category, "prompt": base})
        prompts.append({"id": f"{category}_{i*4+2}", "category": category, "prompt": base + " in the morning light"})
        prompts.append({"id": f"{category}_{i*4+3}", "category": category, "prompt": base + " with cinematic lighting"})
        prompts.append({"id": f"{category}_{i*4+4}", "category": category, "prompt": base + ", high quality, detailed"})

with open('benchmark_dataset.json', 'w') as f:
    json.dump(prompts, f, indent=2)

print(f"Generated {len(prompts)} benchmark prompts in benchmark_dataset.json")
