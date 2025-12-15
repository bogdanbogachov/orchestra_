import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_data", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--infer_default", type=bool, default=False)
    parser.add_argument("--infer_custom", type=bool, default=False)
    return parser

