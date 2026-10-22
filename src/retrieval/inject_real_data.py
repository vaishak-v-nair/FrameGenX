import os
import sys
from pathlib import Path
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

def main():
    milvus_path = Path("E:/framegenx/data/index/milvus.db")
    if not milvus_path.exists():
        print("Milvus DB not found.")
        return

    client = MilvusClient(uri=str(milvus_path))
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")

    real_prompts = [
        "Imagine a fantasy video game scene in a mythical landscape with floating islands, cascading waterfalls, and ancient ruins covered in vines. Add an adventurer holding an ancient sword, standing on a jagged cliff as the sun sets.",
        "A cyberpunk city street at midnight, neon signs reflecting in rain puddles. A lone detective in a trench coat walks down a narrow alleyway. Flying cars in the background, cinematic lighting, 8k resolution.",
        "A photorealistic portrait of an elderly wizard with a long white beard, wearing robes embroidered with stars. He is holding a wooden staff with a glowing blue crystal. Intricate details, 8k resolution, cinematic lighting.",
        "An isometric 3D render of a cozy coffee shop. A barista is making espresso, and a customer is reading a book at a table by the window. Plants hanging from the ceiling, soft warm lighting, clay material aesthetic.",
        "A futuristic spaceship bridge, sleek white and silver design, holographic displays. A captain in a blue uniform points towards a large viewing window showing a glowing purple nebula.",
        "A dense magical forest with giant glowing mushrooms, fairies flying around, and a clear stream running through the center. Sun rays breaking through the canopy, ethereal atmosphere.",
        "A post-apocalyptic wasteland, a rusty modified car driving through the desert kicking up dust. A survivor wearing goggles and a scarf looking out the window.",
        "An epic battle between a fiery red dragon and a knight in shining armor on a stone bridge over a river of lava. Dynamic action shot, high contrast.",
        "A serene Japanese zen garden with a koi pond, cherry blossom trees in full bloom, and a small wooden bridge. Peaceful, harmonious, soft natural lighting.",
        "A bustling medieval marketplace, merchants selling fruits and fabrics from wooden stalls, peasants and nobles walking around. A castle looming in the background under a clear blue sky.",
        "A high fantasy landscape featuring a grand crystal castle perched on top of a snow-capped mountain. A dragon flies in the distance. Magical aura, highly detailed, digital art.",
        "A futuristic cyberpunk hacker room, multiple glowing monitors displaying code. A hacker with glowing cybernetic implants typing furiously. Cluttered desk with energy drinks and cables.",
        "A photorealistic close-up of a majestic lion in the African savanna during golden hour. The lion's mane is glowing in the sunlight. National Geographic style.",
        "A 2D pixel art scene of a retro fantasy RPG town. A brave warrior, a wise mage, and a swift rogue stand near the town fountain. 16-bit graphics, vibrant colors.",
        "A surreal dreamscape with floating clocks melting over bare tree branches, inspired by Salvador Dali. A vast empty desert in the background, soft surreal lighting.",
        "A majestic underwater city enclosed in giant glass domes. Mermaids swimming alongside submarines. Glowing bioluminescent plants, deep blue ocean water.",
        "A steampunk laboratory filled with brass gears, steam pipes, and glowing green chemicals in glass flasks. An eccentric inventor working on a clockwork automaton.",
        "A beautiful elven archer drawing her bow in a dense sunlit forest. She wears green and gold leather armor. Highly detailed face, sharp focus, fantasy illustration.",
        "A dramatic shot of a lone astronaut standing on the surface of Mars, looking at Earth in the distant sky. Red rocky terrain, realistic spacesuit, space exploration.",
        "A cozy interior of a winter cabin. A roaring fireplace, a comfortable armchair with a knitted blanket, and a window showing a snowstorm outside. Warm yellow lighting."
    ]

    dense_embs = dense_model.encode(real_prompts, convert_to_numpy=True).tolist()
    sparse_embs = list(sparse_model.embed(real_prompts))

    collection_name = "FrameGenX_RAG"
    
    # Get max id to avoid collision
    try:
        res = client.query(collection_name=collection_name, filter="id >= 0", output_fields=["id"])
        max_id = max([r["id"] for r in res]) if res else 0
    except Exception as e:
        print(f"Error querying max id: {e}")
        max_id = 0

    data = []
    for i, (prompt, dense, sparse) in enumerate(zip(real_prompts, dense_embs, sparse_embs)):
        sparse_dict = {int(idx): float(val) for idx, val in zip(sparse.indices.tolist(), sparse.values.tolist())}
        data.append({
            "id": max_id + 1 + i,
            "dense": dense,
            "sparse": sparse_dict,
            "caption": prompt,
            "image": "real_data",
            "original_id": f"real_{i}"
        })

    client.insert(collection_name=collection_name, data=data)
    print(f"Successfully injected {len(data)} real fantasy/sci-fi/cyberpunk prompts into Milvus!")

if __name__ == "__main__":
    main()
