# 14_INDEX_AND_RETRIEVAL_ARCHITECTURE.md — Multi-Modal Visual RAG & Knowledge-Grounded Generation Architecture

---

## 1. Executive Summary & Paradigm Shift

Historically, text-to-image (T2I) systems operated under an **input-prompt-centric paradigm**: a user enters an unconstrained natural language prompt, and a monolithic diffusion model attempts to infer all semantic, spatial, and stylistic relationships zero-shot.

FrameGenX shifts the paradigm to an **Index-First, Knowledge-Grounded Query Architecture**, inspired by the **NVIDIA NeMo Framework RAG Pipeline** and state-of-the-art **Multi-Modal Knowledge-Augmented Generation (KAG)** principles.

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                PROMPT-CENTRIC PARADIGM (OLD)                                |
|  [Raw Prompt] ────> [Monolithic Diffusion Sampler] ────> [Unconstrained / Probabilistic]   |
+─────────────────────────────────────────────────────────────────────────────────────────────+
                                              VS
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                         INDEX-FIRST KNOWLEDGE-GROUNDED PARADIGM                             |
|                                                                                             |
|  [PHASE 1: OFFLINE MULTI-MODAL INDEXING]                                                    |
|  Visual Corpora + Layout Blueprints + Style Presets + Concept Catalogs + Failure Memory     |
|                                           │                                                 |
|                                           ▼                                                 |
|               Structured Multi-Modal Knowledge Store (`data/index/`)                        |
|                                                                                             |
|  [PHASE 2: ONLINE QUERY-DRIVEN RETRIEVAL & SYNTHESIS]                                       |
|  User Query ──> [Query Decomposer] ──> [Hybrid Dense-Sparse RAG + Layout Blueprint Match]   |
|                                           │                                                 |
|                                           ▼                                                 |
|                    Grounded ContractV2 + Enforced Structural Plan                           |
|                                           │                                                 |
|                                           ▼                                                 |
|                  Constrained Generation ──> Closed-Loop Verification                        |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. The Offline Multi-Modal Indexing Pipeline

Following NVIDIA NeMo's modular indexing design, the offline indexing stage transforms raw visual and structural corpora into five coordinated knowledge facets:

```
                                 THE 5-FACET KNOWLEDGE INDEX
                                 
   [Visual Corpora & Curated Data]
                 │
                 ├─── 1. Dense Semantic Embeddings  ──> `embeddings.npy` (CLIP ViT L2-norm)
                 │
                 ├─── 2. Sparse Lexical BM25        ──> `bm25_index.json` (Inverted Index & IDF)
                 │
                 ├─── 3. Spatial Layout Blueprints  ──> `layout_blueprints.json` (2D Box Schemas)
                 │
                 ├─── 4. Style & Lighting Presets   ──> `style_presets.json` (Lenses & Lighting)
                 │
                 └─── 5. Concept & Failure Memory   ──> `concept_catalog.json` & `failure_memory.json`
```

### 2.1 Facet 1: Dense Semantic Vector Store (`embeddings.npy`)
- **Backbone**: `openai/clip-vit-base-patch32` (or ViT-L/14).
- **Corpus**: Diverse photographic and compositional captions (COCO val2017 curated subset).
- **Representation**: Float32 $[N, D]$ normalized tensor ($D=512$), stored on disk with zero memory footprint until queried.

### 2.2 Facet 2: Sparse Lexical BM25 Index (`bm25_index.json`)
- **Motivation**: Dense embeddings capture global semantics but often miss exact, discrete nouns (e.g., distinguishing `"fork"` from `"spoon"` or `"apple"` from `"orange"`).
- **Formulation**:
$$\text{IDF}(t) = \ln\left(1 + \frac{N_{\text{docs}} - n(t) + 0.5}{n(t) + 0.5}\right)$$
$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
where $k_1 = 1.5, b = 0.75$.

### 2.3 Facet 3: Spatial Layout Blueprint Catalog (`layout_blueprints.json`)
- **Purpose**: Provides mathematically valid 2D bounding boxes $[ymin, xmin, ymax, xmax]$ for 1-entity, 2-entity horizontal splits, support surfaces (table/ground), triangular clusters, 2x2 quad grids, and 5-object arrays.
- **Benefit**: Replaces heuristic guessing with proven geometric templates that eliminate spatial overlap.

### 2.4 Facet 4: Style & Lighting Presets (`style_presets.json`)
- **Profiles**: `photorealistic`, `cinematic`, `studio_product`, `pixel_art`, `isometric_3d`, `cyberpunk_neon`.
- **Fields**: Specific camera lens parameters (e.g., `"50mm f/2.8"`), lighting environment (e.g., `"softbox studio lighting"`), and targeted negative prompts.

### 2.5 Facet 5: Visual Concept Ontology & Failure Memory
- **Affordance Mapping**: Defines supporting surfaces vs. portable objects, scale ratios, and default attributes.
- **Failure Memory**: Direct lookup of repair strategies for known generative traps (under-counting, over-counting, attribute bleed).

---

## 3. The Online Query-Driven Retrieval Engine

When a query arrives, the system does not immediately invoke diffusion. It executes a multi-modal query pipeline:

```
                            ONLINE QUERY RETRIEVAL FLOW
                            
   [User Query: "Three red apples on a rustic wooden table"]
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. QUERY DECOMPOSITION & ENTITY EXTRACTION                  │
   │    • Entities: 3x apple [red], 1x wooden table [rustic]     │
   │    • Relations: apples standing_on table                    │
   │    • Total Discrete Items: 4                                │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 2. HYBRID DENSE-SPARSE RETRIEVAL (RRF)                      │
   │    • Dense search: Finds semantic photographic captions     │
   │    • Sparse BM25: Asserts exact match on "apple" & "table"  │
   │    • Reciprocal Rank Fusion combines top candidates         │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3. SPATIAL BLUEPRINT & STYLE MATCHING                       │
   │    • Layout Matcher: Retrieves "2_entities_subject_table"   │
   │    • Style Matcher: Retrieves "photorealistic" lens/light   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 4. GROUNDED CONTRACT & SCENE PLAN COMPILATION               │
   │    • Emits ContractV2 with retrieved coordinates            │
   │    • Renders ControlNet Canny/Depth segmentation masks      │
   └─────────────────────────────────────────────────────────────┘
```

### 3.1 Reciprocal Rank Fusion (RRF) Formulation
To combine dense vector scores and sparse BM25 scores without score-distribution distortion:
$$\text{RRF}(d) = \frac{w_{\text{dense}}}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{k_{\text{rrf}} + \text{rank}_{\text{sparse}}(d)}$$
where $k_{\text{rrf}} = 60, w_{\text{dense}} = 0.6, w_{\text{sparse}} = 0.4$.

---

## 4. Integration with Scene Planning and Structural Control

Once knowledge is retrieved:
1. **Mask Generation (`src/scene/planner.py`)**: `ScenePlanner` reads `contract["retrieved_layout"]`. If a valid blueprint exists AND the contract has ≥2 counted entities, it scales normalized $[ymin, xmin, ymax, xmax]$ directly into 512x512 pixel coordinate masks. Scenic prompts (0/1 counted entities) skip masks entirely — forcing a grid onto a panorama paints collages.
2. **Prompt Composition**: The 77-token composer anchors on the **ORIGINAL user prompt** (corrective edits prepended, style tail appended). Retrieved captions are **inspection-only** since 2026-08-24: measured CLIP sentence similarity cannot validate caption relevance (any pair scores ~0.27–0.48), and fusing unvalidated captions injected hallucinations. Only curated blueprints and style presets reach the drawing.
3. **ControlNet Boundary Enforcement**: The binary layout masks are converted to Canny edge guides, forcing the diffusion process to respect the retrieved spatial topology (ControlNet path only; dropped for scenic prompts).

---

## 5. Performance and Resource Footprint

| Retrieval Dimension | Measured Performance on RTX 3050 Laptop |
|---|---|
| **Dense Search Latency** | $\approx 4.2\text{ ms}$ (Numpy exact matrix multiplication) |
| **Sparse BM25 Search Latency** | $\approx 1.8\text{ ms}$ (Inverted index lookup) |
| **Blueprint & Style Match Latency** | $< 0.5\text{ ms}$ (In-memory dictionary lookup) |
| **Total Query Retrieval Time** | **$< 7.0\text{ ms}$** |
| **VRAM Consumption** | **0 MB** (Executes completely on CPU before GPU pipeline loads) |
| **System RAM Footprint** | $\approx 15\text{ MB}$ (Extremely lightweight) |

---

## 6. How to Build & Re-index Corpora

```powershell
# Build standard 1,000-caption multi-modal knowledge index:
python scripts/build_index.py --n 1000

# Build custom large-scale index (e.g., 10,000 captions with batch size 128):
python scripts/build_index.py --n 10000 --batch 128
```
