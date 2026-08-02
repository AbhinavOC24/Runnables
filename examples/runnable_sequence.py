from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from utils.model_setup import get_model

model = get_model()


prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke about {topic}"
)
parser = StrOutputParser()


prompt2 = PromptTemplate(
    input_variables=["joke"],
    template="Explain the following joke \n {joke}"
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
print(chain.invoke({'topic': 'Leetcode'}))

