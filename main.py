from cli.parser import build_parser
from commands.data_processing.preprocess import run_preprocess
from commands.training.finetune import run_finetune
from commands.inference.default import run_infer_default
from commands.inference.custom import run_infer_custom
from config import CONFIG

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    experiment = CONFIG.get('experiment', 'orchestra')

    run_preprocess() if args.preprocess_data else None
    run_finetune() if args.finetune else None
    run_infer_default() if args.infer_default else None
    run_infer_custom() if args.infer_custom else None
