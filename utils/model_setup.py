from langchain_huggingface import HuggingFaceEndpoint
from transformers import GenerationConfig
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import warnings
import transformers
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

def get_model(max_new_tokens: int = 256, device_map: str = None) -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="together",
    max_new_tokens=256
    )
    
    return ChatHuggingFace(llm=llm)