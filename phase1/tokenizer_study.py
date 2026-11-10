"""Compare saved corpus-trained tokenizers on common held-out text; no LM required."""
import argparse
import sqlite3
from pathlib import Path
from src.common import *

def compare(prepared_dirs, output):
    prepared_dirs = list(map(Path, prepared_dirs))
    missing = [str(folder / "manifest.json") for folder in prepared_dirs
               if not (folder / "manifest.json").is_file()]
    if missing:
        raise ValueError("Tokenizer preparation is incomplete: " + ", ".join(missing)
                         + ". Finish src.prepare successfully before comparison. Downloaded raw data alone is not a prepared tokenizer.")
    from src.plotting import pyplot
    plt = pyplot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_ids = None
    for folder in prepared_dirs:
        manifest = read(folder / "manifest.json")
        tok = load_tokenizer(folder / "tokenizer.json")
        db = sqlite3.connect(f"file:{(folder / 'documents.sqlite').resolve()}?mode=ro", uri=True)
        ids, per_source = [], {}
        for source in sorted(manifest["audit"]):
            tokens = words = chars = raw_roundtrips = docs = normalized_roundtrips = 0
            for digest, text in db.execute("SELECT digest,text FROM docs WHERE split='validation' AND source=? ORDER BY digest LIMIT 100",(source,)):
                encoded = tok.encode(text)
                decoded = tok.decode(encoded.ids, skip_special_tokens=False)
                tokens += len(encoded.ids)
                words += len(text.split())
                chars += len(text)
                raw_roundtrips += decoded == text
                normalized_roundtrips += decoded == normalized_text(tok,text)
                docs += 1
                ids.append(digest)
            if not docs or normalized_roundtrips != docs:
                raise ValueError(f"Missing held-out data or failed normalized round trip: {folder}/{source}")
            per_source[source] = dict(documents=docs,tokens=tokens,words=words,characters=chars,
                tokens_per_word=tokens/words,tokens_per_character=tokens/chars,
                raw_roundtrips=raw_roundtrips,normalized_roundtrips=normalized_roundtrips)
        db.close()
        if sample_ids is not None and ids != sample_ids:
            raise ValueError("Tokenizers have different evaluation documents; prepare with identical split/data")
        sample_ids = ids
        prompt_lengths=[len(tok.encode(v["text"]).ids) for v in read(folder/"held_out_prompts.json").values()]
        rows.append({"variant":folder.name,"vocab_size":tok.get_vocab_size(),"per_source":per_source,
            "max_512_word_prompt_tokens":max(prompt_lengths),"tokenizer":manifest["tokenizer"]})
    result={"scope":"tokenizer-only comparison on up to 100 common held-out documents per source",
            "sample_ids_sha256":fingerprint(sample_ids),"variants":rows}
    write_json(output/"tokenizer_comparison.json",result)
    lines=["# Tokenizer study","",result["scope"],"",
           "These measurements describe segmentation and context use; they do not establish downstream model quality.","",
           "| Variant | Actual vocabulary | Source | Tokens/word | Tokens/character | Raw round trips | Normalized round trips |",
           "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for source,v in row["per_source"].items():
            lines.append(f"| {row['variant']} | {row['vocab_size']} | {source} | {v['tokens_per_word']:.4f} | {v['tokens_per_character']:.4f} | {v['raw_roundtrips']}/{v['documents']} | {v['normalized_roundtrips']}/{v['documents']} |")
    (output/"TOKENIZERS.md").write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(9,4))
    for row in rows:
        ax.plot(list(row["per_source"]),[v["tokens_per_word"] for v in row["per_source"].values()],marker="o",label=row["variant"])
    ax.set(ylabel="Tokens per whitespace word",title="Common held-out text; lower means fewer tokens")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"tokenizer_comparison.png",dpi=160)
    plt.close(fig)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",nargs="+",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    compare(a.prepared,a.output)
