import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate   # ← chat format, not raw text
from utils.model_setup import get_model

model = get_model()

# Step 1: Generate a joke about the given topic
# ChatPromptTemplate formats messages as [system, human] — correct for instruct models
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty comedian. Keep responses short and concise."),
    ("human", "Write a short, single joke about {topic}. Just the joke, nothing else."),
])
parser = StrOutputParser()

# Step 2: Explain the joke produced by step 1
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep responses short and concise."),
    ("human", "explain why this joke is funny :\n\n{joke}"),
])

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "clown"})
print(result)
