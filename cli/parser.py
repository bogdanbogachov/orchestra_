import argparse

def str_to_bool(v):
    return str(v).lower() in ('true', '1', 'yes')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_data", type=str_to_bool, default=False)
    parser.add_argument("--finetune", type=str_to_bool, default=False)
    parser.add_argument("--infer_default", type=str_to_bool, default=False)
    parser.add_argument("--infer_custom", type=str_to_bool, default=False)
    parser.add_argument("--evaluate", type=str_to_bool, default=False)
    parser.add_argument("--aggregate_results", type=str_to_bool, default=False)
    parser.add_argument("--global_exp_num", type=int, default=None, 
                       help="Global experiment number to aggregate (e.g., 7, 8). If not specified, aggregates all.")
    return parser
