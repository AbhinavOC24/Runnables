# LangChain Runnables — Learning Notes

## What are Runnables?
Runnables are the core abstraction in LangChain's **LCEL (LangChain Expression Language)**. Any object that implements the `Runnable` interface can be chained with the `|` operator (or explicitly via `RunnableSequence`).

Every Runnable exposes:
- `.invoke(input)` — single synchronous call
- `.batch(inputs)` — parallel calls on a list
- `.stream(input)` — stream output tokens

---

## Types of Runnables

### 1. Task-Specific Runnables
Built for a specific purpose and implement the Runnable interface:
| Runnable | Purpose |
|---|---|
| `PromptTemplate` | Formats input variables into a prompt string |
| `ChatPromptTemplate` | Like `PromptTemplate` but for chat message lists |
| `LLM / ChatModel` | Calls a language model |
| `StrOutputParser` | Parses LLM output to a plain string |
| `JsonOutputParser` | Parses LLM output to JSON |

### 2. Runnable Primitives
General-purpose Runnables that help **compose** other Runnables:
| Primitive | Purpose |
|---|---|
| `RunnableSequence` | Chains Runnables one after another (`A | B | C`) |
| `RunnableParallel` | Runs multiple Runnables in parallel on the same input |
| `RunnableLambda` | Wraps any Python function as a Runnable |
| `RunnablePassthrough` | Passes input unchanged (useful for injecting context) |
| `RunnableBranch` | Conditional routing based on input |

---

## RunnableSequence
`RunnableSequence` chains Runnables so the output of each becomes the input of the next.

```python
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(prompt, model, parser)
result = chain.invoke({"topic": "AI"})
# Equivalent shorthand using LCEL:
chain = prompt | model | parser
```

**Key point:** Each Runnable in the sequence must accept the output type of the previous one.

---

## Common Gotchas
- **Use `ChatPromptTemplate` for instruct/chat models** — `PromptTemplate` produces raw text. Instruct models (like `Qwen2-*-Instruct`) expect structured `[system, user, assistant]` chat messages. Feeding raw text causes the model to hallucinate the full conversation (printing "Human: ... Assistant: ..." in a loop).
- `PromptTemplate` uses `template=` (not `tempalte=`) — typo causes a silent `None` template
- **Set `repetition_penalty > 1.0`** (e.g. `1.3`) in `GenerationConfig` to stop small models from looping once they start repeating tokens
- **Keep `max_new_tokens` low** for small models (≤128) — they don't benefit from more tokens and just start looping
- `HuggingFacePipeline` needs `return_full_text: False` to avoid echoing the prompt in the output

---

## Setup

```bash
# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
