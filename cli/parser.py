import argparse

def str_to_bool(v):
    return str(v).lower() in ('true', '1', 'yes')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_data", type=str_to_bool, default=False)
    parser.add_argument("--add_noise", type=str_to_bool, default=False)
    parser.add_argument("--noise_type", type=str, required=False, default=None, choices=["banking77", "clinc150"],
                        help="Type of noise to use: 'banking77' or 'clinc150' (required when using --add_noise)")
    parser.add_argument("--finetune", type=str_to_bool, default=False)
    parser.add_argument("--infer_default", type=str_to_bool, default=False)
    parser.add_argument("--infer_custom", type=str_to_bool, default=False)
    parser.add_argument("--evaluate", type=str_to_bool, default=False)
    parser.add_argument("--aggregate_results", type=str_to_bool, default=False)
    parser.add_argument("--global_exp_num", type=int, default=None)
    parser.add_argument("--charts", type=str_to_bool, default=False,
                        help="Create F1 scores comparison charts")
    parser.add_argument("--aggregation_nums", type=int, nargs='+', default=None,
                        help="List of aggregation numbers to include in charts (required when using --charts)")
    parser.add_argument("--aggregation_names", type=str, nargs='+', default=None,
                        help="List of custom names for aggregations (must match length of --aggregation_nums)")
    return parser
