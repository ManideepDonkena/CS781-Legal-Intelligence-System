import argparse
from .common import ROOT
from .model_factory import configuration, check_parameters

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, nargs="+", choices=[32000, 64000], default=[32000, 64000])
    args = p.parse_args()
    for vocab in args.vocab_size:
        for path in sorted((ROOT / "configs/arch").glob("*.json")):
            print(path.stem, vocab, check_parameters(configuration(path, vocab, 1)))

if __name__ == "__main__":
    main()
