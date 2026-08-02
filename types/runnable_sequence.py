from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline


llm=HuggingFacePipeline.from_model_id(  
            repo_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            max_new_tokens=512,)
model=ChatHuggingFace(llm=llm)


prompt=PromptTemplate(
    input_variables=["topic"],
    tempalte="Write a joke about {topic}"
)
parser=StrOutputParser()
chain=RunnableSequence(prompt,model,parser)
print(chain.invoke({'topic':'Leetcode'}))
