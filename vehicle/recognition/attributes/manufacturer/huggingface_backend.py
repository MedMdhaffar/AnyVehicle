"""Local Transformers adapter with the same response contract as llama.cpp."""
import io
from pathlib import Path

from .llama_backend import ContextBudgetError, InferenceError


class HuggingFaceBackend:
    """Job-owned GPU model. ManufacturerService serializes every call."""

    def __init__(self, config):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if config.device != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("Recognition requires cuda:0; select the GPU with CUDA_VISIBLE_DEVICES")
        self._torch = torch
        self.config = config
        self.model = self.processor = None
        model_path = str(Path(config.hf_model_path).expanduser().resolve())
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
            ).to(config.device).eval()
        except BaseException:
            self.close()
            raise

    def generate(self, image_bytes, prompt, *, temperature, top_p, max_tokens):
        from PIL import Image

        if self.model is None:
            raise InferenceError("Recognizer is closed")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                inputs = self.processor.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": prompt},
                    ]}], add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(self.model.device, dtype=self.model.dtype)
            prompt_tokens = inputs["input_ids"].shape[-1]
            if prompt_tokens + max_tokens > self.config.context_size:
                raise ContextBudgetError(
                    f"Insufficient context: {prompt_tokens} prompt tokens + {max_tokens} output > {self.config.context_size}"
                )
            options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0, "use_cache": True}
            if temperature > 0:
                options.update(temperature=temperature, top_p=top_p)
            with self._torch.inference_mode():
                output = self.model.generate(**inputs, **options)
            generated = output[:, prompt_tokens:]
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            eos_ids = self.model.generation_config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            stopped = generated.shape[-1] > 0 and generated[0, -1].item() in (eos_ids or [])
            finish = "length" if generated.shape[-1] >= max_tokens and not stopped else "stop"
            return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Hugging Face VLM inference failed: {exc}") from exc

    def close(self):
        self.model = self.processor = None
        with self._torch.cuda.device(self.config.device):
            self._torch.cuda.empty_cache()
