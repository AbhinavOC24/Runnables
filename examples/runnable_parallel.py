import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from utils.model_setup import get_model
from dotenv import load_dotenv
load_dotenv()
model = get_model()
parser = StrOutputParser()

# --- Branch 1: Generate a joke about the topic ---
joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty comedian. Keep responses short and concise."),
    ("human", "Write a short, single joke about {topic}. Just the joke, nothing else."),
])
joke_chain = RunnableSequence(joke_prompt, model, parser)

# --- Branch 2: Generate a fun fact about the topic ---
fact_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable trivia expert. Keep responses short and concise."),
    ("human", "Tell me one surprising fun fact about {topic}. Just the fact, nothing else."),
])
fact_chain = RunnableSequence(fact_prompt, model, parser)

# --- Branch 3: Generate a short poem about the topic ---
poem_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative poet. Keep responses short and concise."),
    ("human", "Write a short 4-line poem about {topic}. Just the poem, nothing else."),
])
poem_chain = RunnableSequence(poem_prompt, model, parser)

# RunnableParallel runs all three chains simultaneously on the SAME input.
# The result is a dict: {"joke": "...", "fun_fact": "...", "poem": "..."}
#
# Two equivalent ways to create a RunnableParallel:
#
# 1. Keyword arguments (used below):
#    parallel_chain = RunnableParallel(joke=joke_chain, fun_fact=fact_chain, poem=poem_chain)
#
# 2. Dict-based syntax:
#    parallel_chain = RunnableParallel({
#        "joke": joke_chain,
#        "fun_fact": fact_chain,
#        "poem": poem_chain,
#    })

parallel_chain = RunnableParallel(
    joke=joke_chain,
    fun_fact=fact_chain,
    poem=poem_chain,
)

result = parallel_chain.invoke({"topic": "cats"})

print("=" * 50)
print("🎭  JOKE:")
print(result["joke"])
print("-" * 50)
print("🧠  FUN FACT:")
print(result["fun_fact"])
print("-" * 50)
print("📝  POEM:")
print(result["poem"])
print("=" * 50)
