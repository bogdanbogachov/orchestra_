import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "orchestra",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    try:
        from config import CONFIG
        logging_config = CONFIG.get('logging', {})
        log_dir = log_dir or logging_config.get('log_dir', 'logs')
        experiment_name = CONFIG.get('experiment', 'orchestra')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{experiment_name}_{timestamp}.log'
    except (ImportError, KeyError, AttributeError):
        log_dir = log_dir or 'logs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_file or f'orchestra_{timestamp}.log'

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    log.addHandler(ch)
    log.addHandler(fh)

    return log

logger = setup_logger()
