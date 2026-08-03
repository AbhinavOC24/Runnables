import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from utils.model_setup import get_model
from dotenv import load_dotenv
load_dotenv()
model = get_model()
parser = StrOutputParser()

# --- What is RunnableLambda? ---
# RunnableLambda wraps any Python function into a Runnable so it can be
# used inside chains (RunnableSequence, RunnableParallel, etc.).
# Use it whenever you need custom logic between LLM steps — formatting,
# filtering, counting, calling an API, etc.

# --- Example 1: Simple post-processing with RunnableLambda ---
# Generate a tweet, then use RunnableLambda to make it uppercase.

tweet_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a social media expert. Keep responses short and concise."),
    ("human", "Write a catchy tweet about {topic}. Just the tweet, nothing else."),
])

upper_case = RunnableLambda(lambda x: x.upper())

chain = RunnableSequence(tweet_prompt, model, parser, upper_case)

result = chain.invoke({"topic": "artificial intelligence"})

# Output looks like:
# "AI ISN'T JUST THE FUTURE — IT'S THE NOW! #ARTIFICIALINTELLIGENCE"
# (the tweet string, converted to uppercase by the lambda)

print("=" * 50)
print("UPPERCASED TWEET (via RunnableLambda):")
print(result)
print("=" * 50)


# --- Example 2: RunnableLambda + RunnableParallel ---
# Generate a tweet, then run two RunnableLambdas in parallel on the output:
# one counts words, the other extracts hashtags.

def count_words(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())

def extract_hashtags(text: str) -> list:
    """Pull out any #hashtags from the text."""
    return [word for word in text.split() if word.startswith("#")]

tweet_chain = RunnableSequence(tweet_prompt, model, parser)

# After the tweet is generated (a string), run both lambdas on it in parallel
analysis_chain = RunnableParallel(
    tweet=RunnablePassthrough(),                    # keep the tweet as-is
    word_count=RunnableLambda(count_words),          # count words
    hashtags=RunnableLambda(extract_hashtags),        # extract hashtags
)

full_chain = RunnableSequence(tweet_chain, analysis_chain)

result2 = full_chain.invoke({"topic": "space exploration"})

# Output looks like:
# {
#   "tweet": "The universe is calling — are you ready to answer? #SpaceExploration",
#   "word_count": 12,                         ← computed by RunnableLambda
#   "hashtags": ["#SpaceExploration"]          ← computed by RunnableLambda
# }

print("\n" + "=" * 50)
print("TWEET:")
print(result2["tweet"])
print("-" * 50)
print(f"WORD COUNT: {result2['word_count']}")
print(f"HASHTAGS:  {result2['hashtags']}")
print("=" * 50)
