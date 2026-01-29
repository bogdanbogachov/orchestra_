from cli.parser import build_parser
from commands.data_processing.preprocess import run_preprocess
from commands.data_processing.add_noise import run_add_noise
from commands.training.finetune import run_finetune
from commands.inference.default import run_infer_default
from commands.inference.custom import run_infer_custom
from commands.evaluation.evaluate import run_evaluation
from commands.evaluation.aggregate_results import run_aggregate_results
from config import CONFIG
from logger_config import logger

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    experiment = CONFIG.get('experiment', 'orchestra')

    if args.preprocess_data:
        logger.info("=" * 100)
        logger.info("STEP: PREPROCESS DATA")
        logger.info("=" * 100)
        run_preprocess()
    
    if args.add_noise:
        if not args.noise_type:
            parser.error("--noise_type is required when using --add_noise")
        logger.info("=" * 100)
        logger.info("STEP: ADD NOISE")
        logger.info("=" * 100)
        run_add_noise(noise_type=args.noise_type)
    
    if args.finetune:
        logger.info("=" * 100)
        logger.info("STEP: FINETUNE")
        logger.info("=" * 100)
        run_finetune()
    
    if args.infer_default:
        logger.info("=" * 100)
        logger.info("STEP: INFER DEFAULT")
        logger.info("=" * 100)
        run_infer_default()
    
    if args.infer_custom:
        logger.info("=" * 100)
        logger.info("STEP: INFER CUSTOM")
        logger.info("=" * 100)
        run_infer_custom()
    
    if args.evaluate:
        logger.info("=" * 100)
        logger.info("STEP: EVALUATE")
        logger.info("=" * 100)
        run_evaluation()
    
    if args.aggregate_results:
        logger.info("=" * 100)
        logger.info("STEP: AGGREGATE RESULTS")
        logger.info("=" * 100)
        run_aggregate_results(global_exp_num=args.global_exp_num)
