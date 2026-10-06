"""Ways to get answers from a language model.

Every backend has the same method:

    backend.chat(conversations) -> list of answer strings

A conversation is a list of messages like {"role": "user", "content": "..."}.
Because they share one method, the rest of the code doesn't care where the model runs.

Choose a backend with a short spec string:
    dummy                              fake answers, for tests
    hf:Qwen/Qwen3-4B-Instruct-2507     Hugging Face model on a GPU (Colab)
    hf4bit:Qwen/Qwen2.5-7B-Instruct    the same, loaded in 4-bit to fit bigger models
    ollama:llama3.2:3b                 a model served by Ollama on your own computer
    api:gpt-4o-mini                    any OpenAI-compatible API (needs ADVICELINE_API_KEY)
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    """Remove hidden 'thinking' blocks some models add, and trim spaces."""
    return THINK_RE.sub("", text).strip()


class DummyBackend:
    """Answers instantly without a model. Used by the tests."""

    def __init__(self, reply: str = "Here is some general information."):
        self.name = "dummy"
        self.reply = reply

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        out = []
        for conv in conversations:
            turn = sum(m["role"] == "user" for m in conv)
            out.append(f"{self.reply} (turn {turn})")
        return out


class HFBackend:
    """A Hugging Face model running on this machine's GPU (use it in Colab)."""

    def __init__(self, model_id: str, load_in_4bit: bool = False, batch_size: int = 16,
                 max_new_tokens: int = 512):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.torch = torch
        self.batch_size = batch_size

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"  # decoder models need padding on the left for batched generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        else:
            # transformers 4.56+ calls this argument "dtype"; older versions call it "torch_dtype"
            major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
            kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

        # Greedy decoding (no randomness) so the benchmark can be repeated
        self.gen_config = copy.deepcopy(self.model.generation_config)
        self.gen_config.do_sample = False
        self.gen_config.temperature = None
        self.gen_config.top_p = None
        self.gen_config.top_k = None
        self.gen_config.max_new_tokens = max_new_tokens
        self.gen_config.pad_token_id = self.tok.pad_token_id

    def _render(self, conversation: list[dict]) -> str:
        # enable_thinking=False switches off "thinking" mode in models that have it (Qwen3, SmolLM3);
        # other models simply ignore it
        return self.tok.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        texts = [self._render(c) for c in conversations]
        # Batch prompts of similar length together so there's less padding
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        answers = [""] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                output = self.model.generate(**enc, generation_config=self.gen_config)
            new_tokens = output[:, enc["input_ids"].shape[1]:]
            for i, ids in zip(idx, new_tokens):
                answers[i] = clean(self.tok.decode(ids, skip_special_tokens=True))
        return answers


def _post_json(url: str, payload: dict, headers: dict, retries: int = 5) -> dict:
    """POST JSON and return the JSON reply, waiting and retrying on rate limits or server errors."""
    data = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    raise RuntimeError("unreachable")


class OllamaBackend:
    """A model served by Ollama (https://ollama.com) on this computer."""

    def __init__(self, model: str, host: str = "http://localhost:11434", max_new_tokens: int = 512):
        self.name = f"ollama/{model}"
        self.model = model
        self.url = f"{host}/api/chat"
        self.max_new_tokens = max_new_tokens

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        out = []
        for conv in conversations:
            reply = _post_json(self.url, {
                "model": self.model, "messages": conv, "stream": False,
                "options": {"temperature": 0, "num_predict": self.max_new_tokens},
            }, headers={})
            out.append(clean(reply["message"]["content"]))
        return out


class APIBackend:
    """Any OpenAI-compatible chat API. Set ADVICELINE_API_KEY (and ADVICELINE_API_BASE for non-OpenAI providers).

    Set a spending limit with your provider before running the full benchmark.
    """

    def __init__(self, model: str, max_new_tokens: int = 512):
        key = os.environ.get("ADVICELINE_API_KEY")
        if not key:
            raise RuntimeError("Set the ADVICELINE_API_KEY environment variable first.")
        self.name = f"api/{model}"
        self.model = model
        self.url = os.environ.get("ADVICELINE_API_BASE", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {key}"}
        self.max_new_tokens = max_new_tokens

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        out = []
        for conv in conversations:
            reply = _post_json(self.url, {
                "model": self.model, "messages": conv,
                "temperature": 0, "max_tokens": self.max_new_tokens,
            }, headers=self.headers)
            out.append(clean(reply["choices"][0]["message"]["content"] or ""))
        return out


def make_backend(spec: str, **kwargs):
    """Turn a spec string such as 'hf:Qwen/Qwen3-4B-Instruct-2507' into a backend."""
    kind, _, model = spec.partition(":")
    if kind == "dummy":
        return DummyBackend()
    if kind == "hf":
        return HFBackend(model, **kwargs)
    if kind == "hf4bit":
        return HFBackend(model, load_in_4bit=True, **kwargs)
    if kind == "ollama":
        return OllamaBackend(model)
    if kind == "api":
        return APIBackend(model)
    raise ValueError(f"Unknown backend '{kind}'. Use dummy, hf, hf4bit, ollama or api.")


def slug(spec: str) -> str:
    """A filename-safe version of a model spec: 'hf:Qwen/Qwen3-4B' -> 'hf_Qwen_Qwen3-4B'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec).strip("_")
