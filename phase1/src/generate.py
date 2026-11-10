"""Preserve every prompt token; stop only after 128 complete output words."""
import argparse
import time
import zipfile
import torch
from transformers import AutoConfig
from .common import *
from .model_factory import build


def read_prompts(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate ID: {key}")
            result[key] = value
        return result
    values = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if isinstance(values, list):
        values = unique([(str(row.get("id", i)), row) for i, row in enumerate(values)])
    if not isinstance(values, dict) or not values:
        raise ValueError("Expected nonempty prompt mapping/list")
    for key, row in values.items():
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or len(row["text"].split()) != 512:
            raise ValueError(f"Prompt {key} must contain exactly 512 words")
    return values


@torch.inference_mode()
def generate(model, tok, prompt, device, max_tokens=1024, temperature=0.1, top_p=0.95):
    prompt_ids = tok.encode(prompt).ids
    context = model.config.max_position_embeddings
    if len(prompt_ids) >= context:
        raise ValueError(f"Prompt has {len(prompt_ids)} tokens, context={context}; no truncation allowed")
    ids = torch.tensor([prompt_ids], device=device)
    emitted, past = [], None
    forbidden = set(getattr(tok, "special_ids", [])) | {tok.token_to_id("<|endoftext|>")}
    for _ in range(min(max_tokens, context - len(prompt_ids))):
        result = model(input_ids=ids, past_key_values=past, use_cache=True)
        past = result.past_key_values
        logits = result.logits[0, -1].float()
        for index in forbidden:
            if index is not None:
                logits[index] = -float("inf")
        if temperature == 0:
            token = logits.argmax()
        else:
            probs, order = (logits / temperature).softmax(-1).sort(descending=True)
            probs[probs.cumsum(-1) - probs > top_p] = 0
            token = order[torch.multinomial(probs, 1)].squeeze()
        emitted.append(int(token))
        text = tok.decode(emitted, skip_special_tokens=False)
        words = text.split()
        if len(words) >= 129 or (len(words) >= 128 and text[-1:].isspace()):
            return " ".join(words[:128])
        ids = token.reshape(1, 1)
    raise ValueError("Could not generate 128 complete words within context/token budget; no padded output")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.max_tokens < 1 or a.temperature < 0 or not 0 < a.top_p <= 1:
        raise ValueError("Invalid generation settings")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    folder = Path(a.model)
    provenance = read(folder / "provenance.json")
    if digest_file(folder / "tokenizer.json") != provenance["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match checkpoint")
    tok = load_tokenizer(folder / "tokenizer.json")
    cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
    model = build(cfg)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.to(a.device).eval()
    prompts = read_prompts(a.prompts)
    start = time.monotonic()
    predictions = {}
    for key, value in prompts.items():
        predictions[key] = {"text": generate(model, tok, value["text"], a.device, a.max_tokens, a.temperature, a.top_p)}
        print(f"generated {len(predictions)}/{len(prompts)}", flush=True)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prediction.json", predictions)
    with zipfile.ZipFile(output / "submission.zip.tmp", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "prediction.json", "prediction.json")
    (output / "submission.zip.tmp").replace(output / "submission.zip")
    from .submission import validate
    write_json(output / "submission_validation.json", validate(a.prompts, output / "submission.zip"))
    write_json(output / "generation.json", {"settings": vars(a), "examples": len(predictions),
        "prompt_sha256": digest_file(a.prompts), "model_sha256": digest_file(folder / "model.pt"),
        "prediction_sha256": digest_file(output / "prediction.json"), "wall_seconds": time.monotonic() - start})

if __name__ == "__main__":
    main()
