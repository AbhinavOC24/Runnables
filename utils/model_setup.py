from transformers import GenerationConfig
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

def get_model(max_new_tokens: int = 256, device_map: str = None) -> ChatHuggingFace:
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2-0.5B-Instruct",
        task='text-generation',
        device_map=device_map,
        pipeline_kwargs={
            "return_full_text": False,
        },
        model_kwargs={
            "generation_config": GenerationConfig(max_new_tokens=max_new_tokens)
        },
    )
    return ChatHuggingFace(llm=llm)