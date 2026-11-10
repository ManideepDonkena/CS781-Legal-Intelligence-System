import torch
from transformers import AutoConfig, AutoModelForCausalLM
from .common import ROOT, read

CLASSES = {"llama": "LlamaDecoderLayer", "gpt_neox": "GPTNeoXLayer",
           "falcon": "FalconDecoderLayer", "mixtral": "MixtralDecoderLayer"}


def configuration(arch, vocab_size, eos_id, context=2048, tiny=False):
    path = Path(arch) if str(arch).endswith(".json") else ROOT / "configs" / "arch" / f"{arch}.json"
    values = read(path)
    kind = values.pop("model_type")
    values.update(vocab_size=vocab_size, eos_token_id=eos_id, bos_token_id=None,
                  pad_token_id=None, max_position_embeddings=context, use_cache=False)
    if tiny:
        values.update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, num_kv_heads=2)
        if kind == "mixtral":
            values.update(num_local_experts=4, num_experts_per_tok=2)
    return AutoConfig.for_model(kind, **values)


def build(config):
    return AutoModelForCausalLM.from_config(config, attn_implementation="eager")


def parameter_count(config):
    with torch.device("meta"):
        model = build(config)
    return sum(p.numel() for p in model.parameters())


def check_parameters(config, tiny=False):
    count = parameter_count(config)
    if count > 300_000_000 or (not tiny and count < 50_000_000):
        raise ValueError(f"Parameter gate: {count:,}; require 50M–300M for production")
    return count


from pathlib import Path
