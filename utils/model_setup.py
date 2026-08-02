from transformers import GenerationConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def get_model(max_new_tokens: int = 128, device_map: str = None) -> ChatHuggingFace:
    """
    Load Qwen2-0.5B-Instruct as a ChatHuggingFace Runnable.

    max_new_tokens: kept low (128) to avoid runaway generation on small models.
    repetition_penalty: penalises the model for repeating the same tokens — fixes looping.
    do_sample / temperature: adds variety so the model doesn't greedily repeat itself.
    """
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.3,   # >1.0 penalises repeated tokens; fixes looping
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2-0.5B-Instruct",
        task="text-generation",
        device_map=device_map,
        pipeline_kwargs={
            "return_full_text": False,  # don't echo the prompt in the output
        },
        model_kwargs={
            "generation_config": generation_config,
        },
    )
    return ChatHuggingFace(llm=llm)