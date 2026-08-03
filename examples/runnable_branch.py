import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from utils.model_setup import get_model
from dotenv import load_dotenv
load_dotenv()
model = get_model()
parser = StrOutputParser()

# --- What is RunnableBranch? ---
# RunnableBranch is the if/else of Runnables.
#
#   if condition_1(input):   run chain_1
#   elif condition_2(input): run chain_2
#   else:                    run default_chain

# Three styles: joke, poem, or a plain explanation (default)
joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian. Respond only with a short joke."),
    ("human", "Tell me a joke about {topic}"),
])

poem_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a poet. Respond only with a short 4-line poem."),
    ("human", "Write a poem about {topic}"),
])

default_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep it to one sentence."),
    ("human", "Explain {topic}"),
])

# Conditions check the "style" key in the input
branch_chain = RunnableBranch(
    (RunnableLambda(lambda x: x["style"] == "joke"), RunnableSequence(joke_prompt, model, parser)),
    (RunnableLambda(lambda x: x["style"] == "poem"), RunnableSequence(poem_prompt, model, parser)),
    RunnableSequence(default_prompt, model, parser),  # default
)

# Output looks like:
# style="joke" -> comedian chain   -> "Why did the cat sit on the computer?..."
# style="poem" -> poet chain       -> "Roses are red..."
# style="other"-> default chain    -> "Cats are domesticated mammals..."

inputs = [
    {"topic": "cats", "style": "joke"},
    {"topic": "cats", "style": "poem"},
    {"topic": "cats", "style": "explain"},
]

for inp in inputs:
    result = branch_chain.invoke(inp)
    print("=" * 50)
    print(f"TOPIC: {inp['topic']}  |  STYLE: {inp['style']}")
    print(f"RESPONSE: {result}")
    print("=" * 50)
    print()
