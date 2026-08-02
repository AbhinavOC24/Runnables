from transformers import GenerationConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def get_model(max_new_tokens: int = 128, device_map: str = None) -> ChatHuggingFace:
    """
    Load Qwen2.5-1.5B-Instruct as a ChatHuggingFace Runnable.

    Why 1.5B over 0.5B?
    - 0.5B doesn't reliably follow instructions — it hallucinates random
      training-data patterns (MCQ answers, essays, etc.) instead of the prompt.
    - 1.5B is the minimum size where Qwen2.5-Instruct stays on topic.
    - Still small enough to run on CPU / MacBook Air.

    max_new_tokens: kept at 128 to avoid runaway generation.
    repetition_penalty: penalises repeated tokens — fixes looping on small models.
    do_sample / temperature: adds variety so the model doesn't greedily repeat itself.
    """
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",   # upgraded from 0.5B
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