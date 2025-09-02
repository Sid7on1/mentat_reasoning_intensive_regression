import openai
from transformers import GPTModel, GPTTokenizer
import tqdm
import asyncio
import logging
import time
import random
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "llm_model": "gpt-4.1",  # or "gpt-5"
    "max_tokens": 1024,
    "temperature": 0.9,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop_sequences": ["\n", "<|endoftext|>"],
    "max_retries": 5,
    "retry_delay": 2,
    "rate_limit_sleep": 60,
}

# Exception classes
class LLMResponseError(Exception):
    pass

class RateLimitedError(Exception):
    pass


# Main LLM interface class
class LLMInterface:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self._get_llm_model()
        self.tokenizer = self._get_llm_tokenizer()
        self.max_batch_size = self.model.config.max_position_embeddings // 2

    def _get_llm_model(self) -> GPTModel:
        model_name = "openai-" + config["llm_model"]
        logger.info(f"Initializing LLM model: {model_name}")
        return GPTModel.from_pretrained(model_name)

    def _get_llm_tokenizer(self) -> GPTTokenizer:
        model_name = "openai-" + config["llm_model"]
        logger.info(f"Initializing LLM tokenizer: {model_name}")
        return GPTTokenizer.from_pretrained(model_name)

    def _encode_inputs(self, prompts: list[str]) -> dict:
        """Encodes a list of prompts into a format suitable for LLM prediction."""
        encoded_inputs = self.tokenizer.batch_encode_plus(
            prompts,
            max_length=config["max_tokens"],
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        return encoded_inputs

    def _decode_responses(self, outputs: dict) -> list[str]:
        """Decodes LLM outputs into a list of text responses."""
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs["generated_tokens"]
        ]
        return responses

    def _handle_rate_limits(self, response: dict) -> None:
        """Handles rate limit errors from the OpenAI API."""
        if response.get("model") and response["model"].get("object") == "rate_limit":
            logger.warning("Rate limited by OpenAI API. Sleeping for 1 minute.")
            time.sleep(config["rate_limit_sleep"])
            raise RateLimitedError("Rate limited by OpenAI API.")

    def _retry_on_error(self, func, *args, **kwargs):
        """Retries a function call that raises specific errors."""
        max_retries = config["max_retries"]
        retry_delay = config["retry_delay"]
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (LLMResponseError, RateLimitedError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Error occurred. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}. Error: {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Maximum retries reached. Error: {e}")
                    raise

    def get_predictions(self, prompts: list[str], num_responses: int = 1) -> list[str]:
        """Gets predictions from the LLM for a list of prompts."""
        encoded_inputs = self._encode_inputs(prompts)
        batch_size = min(len(prompts), self.max_batch_size)

        responses = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            batch_inputs = {
                k: v[i : i + batch_size].cuda() if isinstance(v, torch.Tensor) else v[i : i + batch_size]
                for k, v in encoded_inputs.items()
            }

            def predict_batch():
                outputs = self.model.generate(
                    **batch_inputs,
                    num_return_sequences=num_responses,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    frequency_penalty=config["frequency_penalty"],
                    presence_penalty=config["presence_penalty"],
                    max_length=config["max_tokens"],
                    stop_sequences=config["stop_sequences"],
                )
                return self._decode_responses(outputs.cpu().numpy())

            responses.extend(self._retry_on_error(predict_batch))

        return responses

    def generate_multiple_rollouts(self, prompt: str, num_rollouts: int) -> list[str]:
        """Generates multiple rollouts for a single prompt."""
        responses = self.get_predictions([prompt] * num_rollouts)
        return responses

    async def async_batch_predict(self, prompts: list[str]) -> list[str]:
        """Uses asynchronous batch prediction for improved performance."""
        responses = await asyncio.gather(*[asyncio.to_thread(self.get_predictions, [prompt]) for prompt in prompts])
        responses = [response for sublist in responses for response in sublist]
        return responses

# Example usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    else:
        llm = LLMInterface(api_key)
        prompts = [
            "What is the square root of 16?",
            "Write a concise summary of this text: The quick brown fox jumps over the lazy dog.",
        ]
        responses = llm.get_predictions(prompts, num_responses=3)
        for prompt, response in zip(prompts, responses):
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}\n")