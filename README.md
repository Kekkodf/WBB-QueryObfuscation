# Words Blending Boxes (WBB) – Obfuscating Queries in Information Retrieval using Differential Privacy 📦

**Words Blending Boxes (WBB)** is a novel mechanism to obfuscate natural‑language queries under **ε‑Differential Privacy**, designed specifically for non‑cooperative information retrieval (IR) systems. WBB ensures both **formal privacy guarantees** and **practical obfuscation** by excluding semantically similar words from candidate sets, while still preserving retrieval utility.

---

## 🚀 Key Features

- **Safe Box**: All top-**k** nearest neighbors (most semantically similar words) to each original query term are excluded.
- **Candidate Box**: A set of **n** words outside the safe box used for substitution.
- **Exponential Mechanism**: Samples replacements with utility based on normalized embedding similarity (Z‑score), preserving ε‑DP guarantees with sensitivity bounded to 1.

---

## 📦 Repo Structure

```
/
├── src/                # Core implementation (tokenization, embeddings, WBB algorithm)
├── results/privacy     # Metrics: lexical/semantic similarity, recall, nDCG@10
├── environment.yml     # Python environment specification
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ⚙️ Installation & Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate wbb

# Or using pip
pip install -r requirements.txt
```

Dependencies include:
- Non-contextual word embeddings (e.g. GloVe or FastText)
- POS-tagging tool (e.g. spaCy or NLTK)
- IR collection loaders (e.g. TREC Robust ’04, MSMARCO)
- Search tools (BM25, TF‑IDF vector space, TAS‑B, Contriever)

---

## 🧠 Workflow Description

1. **Pre‑processing**:
   - Normalize query (lowercase, tokenize, POS-tag).
   - Extract target words (e.g. nouns/adjectives or sensitive terms).
   - Map each to fixed non-contextual embeddings IR-compatible.

2. **Mapping (fₖ,ₙ)_mapping**:
   - Compute semantic similarity (cosine, Euclidean, or product).
   - Exclude top‑k nearest neighbor words → *Safe Box*.
   - From words ranked outside that threshold, select top‑n → *Candidate Box*.

3. **Sampling (fₖ,ₙ,ε)_sampling**:
   - Compute utility score u(w’) = normalized similarity (Z‑score) with original term.
   - Use exponential mechanism to sample obfuscated word from candidate box under budget ε.
   - Guarantees ε‑DP due to sensitivity = 1 and bounded candidate set size.

---

## 📈 Evaluation & Results

- **Privacy Metrics**: Lexical and semantic similarity between original and obfuscated queries is low, showing effective obfuscation.
- **Utility Metrics**:
  - **Recall**: Aggregates retrieved documents across obfuscated queries → high coverage on TREC Robust ’04 and DL’19 (MSMARCO).
  - **nDCG@10**: After reranking with original query locally on retrieved results—utility remains competitive.
- Comparison with alternatives:
  - Outperforms heuristic methods (hypernyms, co-occurrence substitutions).
  - Offers tunable privacy–utility trade‑off by varying ε, k, n.

---

## 🔬 Tuning Parameters

- **k (safe box size)**: Larger → higher privacy, fewer close terms allowed.
- **n (candidate box size)**: Larger → more diverse word space → higher utility (but slower).
- **ε (privacy budget)**: Lower → stronger privacy (more randomness); higher → better utility.  
Tuning these helps balance privacy vs. effectiveness based on user need.

---

## 🧩 Embeddings & Similarity Choices

- Supports multiple similarity modes:
  - *Cosine similarity* (“angle obfuscation”)
  - *Euclidean distance* (“distance obfuscation”)
  - *Product* of both
- Users can swap GloVe, FastText, or any pre‑trained non-contextual embeddings.

