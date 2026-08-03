# LangChain Runnables — Learning Repo

A hands-on learning repository for exploring **LangChain Runnables** and **LCEL (LangChain Expression Language)**.  
The goal is to build intuition for how LangChain's composable pipeline primitives work by writing and running small, focused examples.

---

## What This Repo Covers

- **Runnable primitives** — `RunnableSequence`, `RunnableParallel`, `RunnableLambda`, `RunnablePassthrough`
- **Task-specific Runnables** — `PromptTemplate`, `ChatModel`, `OutputParser`
- **Local LLMs via HuggingFace** — running small models (e.g. Qwen2-0.5B) locally with `langchain-huggingface`
- **LCEL pipelines** — composing chains using the `|` operator and explicit `RunnableSequence`

---

## Project Structure

```
Runnables/
├── examples/               # Runnable example scripts (one concept per file)
│   ├── runnable_sequence.py
│   └── runnable_parallel.py
├── utils/
│   └── model_setup.py      # Shared model loader (HuggingFacePipeline → ChatHuggingFace)
├── NOTES.md                # Concept notes and key learnings
├── requirements.txt        # Pinned Python dependencies
└── venv/                   # Virtual environment (git-ignored)
```

---

## Getting Started

### 1. Clone & Set Up

```bash
git clone <your-repo-url>
cd Runnables

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run an Example

```bash
# Run from the repo root so relative imports resolve correctly
python3 -m examples.runnable_sequence
```

---

## Examples

| File | Concept Demonstrated |
|---|---|
| [`examples/runnable_sequence.py`](examples/runnable_sequence.py) | Chaining prompts + model + parser using `RunnableSequence` |
| [`examples/runnable_parallel.py`](examples/runnable_parallel.py) | Running multiple chains simultaneously using `RunnableParallel` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-core` | Core Runnable abstractions and LCEL |
| `langchain-huggingface` | HuggingFace model integration |
| `transformers` | Local model loading |

---

## Notes

See [`NOTES.md`](NOTES.md) for detailed concept notes, a breakdown of Runnable types, and common gotchas encountered during learning.
