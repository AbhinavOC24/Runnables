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
| --- | --- |
| `PromptTemplate` | Formats input variables into a prompt string |
| `ChatPromptTemplate` | Like `PromptTemplate` but for chat message lists |
| `LLM / ChatModel` | Calls a language model |
| `StrOutputParser` | Parses LLM output to a plain string |
| `JsonOutputParser` | Parses LLM output to JSON |

### 2. Runnable Primitives

General-purpose Runnables that help **compose** other Runnables:

| Primitive | Purpose |
| --- | --- |
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

## RunnableParallel

`RunnableParallel` runs multiple Runnables **simultaneously** on the **same input** and returns a dict of their results.

```python
from langchain_core.runnables import RunnableParallel, RunnableSequence

joke_chain  = RunnableSequence(joke_prompt, model, parser)
fact_chain   = RunnableSequence(fact_prompt, model, parser)

# Method 1: Keyword arguments
parallel = RunnableParallel(joke=joke_chain, fact=fact_chain)

# Method 2: Dict-based syntax (equivalent)
parallel = RunnableParallel({
    "joke": joke_chain,
    "fact": fact_chain,
})

result = parallel.invoke({"topic": "cats"})
# → {"joke": "Why did the cat sit on the computer? ...", "fact": "Cats sleep 12-16 hours ..."}
```

**Key points:**
- Every branch receives the **same input** (not the output of another branch).
- The result is always a **dict** keyed by the names you assigned.
- Branches run in **concurrent threads**, so total latency ≈ the slowest branch (not the sum).

---

## RunnablePassthrough

`RunnablePassthrough` forwards its input unchanged. It's most useful inside `RunnableParallel` to preserve the original input alongside transformed outputs.

### Basic usage inside RunnableParallel

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence

tweet_chain = RunnableSequence(tweet_prompt, model, parser)

chain = RunnableParallel(
    original_input=RunnablePassthrough(),   # forwards {"topic": "AI"} as-is
    tweet=tweet_chain,                      # generates a tweet from the topic
)

result = chain.invoke({"topic": "AI"})
# → {"original_input": {"topic": "AI"}, "tweet": "🤖 AI is transforming ..."}
```

### `RunnablePassthrough.assign()`

`.assign()` is a shortcut that passes **all existing keys through** and adds new computed keys on top — no need to manually forward each key with `RunnablePassthrough()`.

```python
summary_chain = RunnableSequence(summary_prompt, model, parser)

chain = RunnablePassthrough.assign(summary=summary_chain)

result = chain.invoke({"topic": "quantum computing"})
# → {"topic": "quantum computing", "summary": "Quantum computing uses ..."}
```

**Key points:**
- `RunnablePassthrough()` alone simply returns its input — useful as a "no-op" branch.
- `.assign(key=runnable)` keeps every original key and **adds** `key` with the Runnable's output.
- Common pattern: pair with `RunnableParallel` to send the original input + an LLM result to the next step in a chain.
---

## RunnableLambda

`RunnableLambda` wraps any Python function into a Runnable, letting you insert custom logic anywhere in a chain.

### In a sequence (post-processing)

```python
from langchain_core.runnables import RunnableLambda, RunnableSequence

upper_case = RunnableLambda(lambda x: x.upper())

chain = RunnableSequence(prompt, model, parser, upper_case)
result = chain.invoke({"topic": "AI"})
# → "🤖 AI ISN'T JUST THE FUTURE — IT'S THE NOW!"
```

### Named functions (better for readability & debugging)

```python
def count_words(text: str) -> int:
    return len(text.split())

def extract_hashtags(text: str) -> list:
    return [w for w in text.split() if w.startswith("#")]

word_counter = RunnableLambda(count_words)
hashtag_extractor = RunnableLambda(extract_hashtags)
```

### Combined with RunnableParallel

```python
analysis = RunnableParallel(
    tweet=RunnablePassthrough(),
    word_count=RunnableLambda(count_words),
    hashtags=RunnableLambda(extract_hashtags),
)

chain = RunnableSequence(tweet_chain, analysis)
result = chain.invoke({"topic": "space"})
# → {"tweet": "🚀 ...", "word_count": 12, "hashtags": ["#Space"]}
```

**Key points:**
- Any callable `f(input) -> output` can be wrapped — lambdas, regular functions, even class methods.
- The function's **return type becomes the next step's input type**, so make sure types match.
- Named functions are preferred over lambdas for clarity in error tracebacks and LangSmith traces.

---

## RunnableBranch

`RunnableBranch` is the **if/else of Runnables**. It evaluates a list of `(condition, runnable)` pairs in order and runs the first one whose condition returns `True`. If none match, it runs a default runnable.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

def is_science(input: dict) -> bool:
    return "atom" in input["question"].lower()

def is_history(input: dict) -> bool:
    return "war" in input["question"].lower()

branch = RunnableBranch(
    (RunnableLambda(is_science), science_chain),   # if is_science → science_chain
    (RunnableLambda(is_history), history_chain),    # elif is_history → history_chain
    general_chain,                                  # else → general_chain (default)
)

result = branch.invoke({"question": "Why do atoms bond?"})
# → routes to science_chain
```

**Think of it as:**
```python
if is_science(input):    return science_chain.invoke(input)
elif is_history(input):  return history_chain.invoke(input)
else:                    return general_chain.invoke(input)
```

**Key points:**
- Conditions are checked **in order** — the first `True` wins.
- The **last argument** (no condition) is the default / `else` branch.
- Conditions must be Runnables — wrap plain functions with `RunnableLambda`.
- Each branch receives the **same original input**.

---

### `PromptTemplate`
Produces a **plain string**. Good for base/completion models that do open-ended text continuation.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke about {topic}"
)
prompt.invoke({"topic": "AI"})
# → StringPromptValue: "Write a joke about AI"
```

### `ChatPromptTemplate.from_messages()`
Produces a **list of structured messages** (`[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`). Required for instruct/chat-tuned models.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty comedian."),
    ("human", "Write a short joke about {topic}."),
])
prompt.invoke({"topic": "AI"})
# → ChatPromptValue: [SystemMessage(...), HumanMessage(...)]
```

### Why `from_messages()` for instruct models?

Instruct models (e.g. `Qwen2.5-*-Instruct`, `Llama-3-Instruct`) are **fine-tuned on chat transcripts**, not raw text. During fine-tuning, every example had explicit role tags (`<|system|>`, `<|user|>`, `<|assistant|>`). At inference time, the tokenizer's `apply_chat_template()` converts the structured messages back into these special tokens.

| | `PromptTemplate` | `ChatPromptTemplate.from_messages()` |
|---|---|---|
| Output type | `StringPromptValue` (raw string) | `ChatPromptValue` (list of messages) |
| Model type | Base / completion models | Instruct / chat-tuned models ✓ |
| Role semantics | None — model guesses | Explicit system / user / assistant |
| What the model receives | `"Write a joke about AI"` | `<\|system\|>You are...<\|user\|>Write a joke...` |
| Risk if wrong format | Model hallucinates conversation | — |

> **Rule of thumb:** If the model name contains `-Instruct`, `-Chat`, or `-it`, always use `ChatPromptTemplate`.



## Common Gotchas

- **Use `ChatPromptTemplate` for instruct/chat models** — `PromptTemplate` produces raw text. Instruct models (like `Qwen2-*-Instruct`) expect structured `[system, user, assistant]` chat messages. Feeding raw text causes the model to hallucinate the full conversation (printing "Human: ... Assistant: ..." in a loop).
- **`0.5B` models cannot reliably follow instructions** — they hallucinate random training-data patterns (MCQ answers, essays, system design docs) instead of responding to the prompt. `1.5B` is the practical minimum for `Qwen2.5-Instruct` to stay on topic.
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
