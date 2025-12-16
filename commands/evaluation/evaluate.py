import json
import time
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from statsmodels.stats.contingency_tables import mcnemar
from config import CONFIG
from logger_config import logger
from commands.inference.default import run_infer_default
from commands.inference.custom import run_infer_custom

def load_test_data():
    paths_config = CONFIG['paths']
    test_file = paths_config['data']['test']
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    return texts, labels

def evaluate_model(inference_fn, texts, labels, model_name):
    predictions = []
    latencies = []
    
    logger.info(f"Evaluating {model_name} on {len(texts)} test samples...")
    
    for i, text in enumerate(texts):
        start_time = time.time()
        result = inference_fn(text)
        end_time = time.time()
        
        pred = result['probs'].argmax(dim=-1).item()
        predictions.append(pred)
        latencies.append((end_time - start_time) * 1000)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(texts)} samples")
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    logger.info(f"✓ {model_name} evaluation complete")
    
    return {
        'predictions': predictions,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'latencies': latencies
    }

def mcnemar_test(y_true, y_pred1, y_pred2):
    both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    both_wrong = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    model1_correct_model2_wrong = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    model1_wrong_model2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    contingency_table = np.array([
        [both_correct, model1_correct_model2_wrong],
        [model1_wrong_model2_correct, both_wrong]
    ])
    
    if model1_correct_model2_wrong + model1_wrong_model2_correct < 25:
        result = mcnemar(contingency_table, exact=True, correction=False)
    else:
        result = mcnemar(contingency_table, exact=False, correction=True)
    
    return result, contingency_table

def run_evaluation():
    logger.info("Starting evaluation pipeline...")
    
    paths_config = CONFIG['paths']
    experiment_name = CONFIG.get('experiment', 'orchestra')
    experiments_dir = paths_config['experiments']
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    texts, labels = load_test_data()
    logger.info(f"Loaded {len(texts)} test samples")
    
    default_results = evaluate_model(run_infer_default, texts, labels, "Default Head")
    custom_results = evaluate_model(run_infer_custom, texts, labels, "Custom Head")
    
    mcnemar_result, contingency_table = mcnemar_test(
        np.array(labels),
        default_results['predictions'],
        custom_results['predictions']
    )
    
    results = {
        'experiment': experiment_name,
        'test_samples': len(texts),
        'default_head': {
            'accuracy': float(default_results['accuracy']),
            'precision': float(default_results['precision']),
            'recall': float(default_results['recall']),
            'f1': float(default_results['f1']),
            'avg_latency_ms': float(default_results['avg_latency_ms']),
            'std_latency_ms': float(default_results['std_latency_ms'])
        },
        'custom_head': {
            'accuracy': float(custom_results['accuracy']),
            'precision': float(custom_results['precision']),
            'recall': float(custom_results['recall']),
            'f1': float(custom_results['f1']),
            'avg_latency_ms': float(custom_results['avg_latency_ms']),
            'std_latency_ms': float(custom_results['std_latency_ms'])
        },
        'mcnemar_test': {
            'statistic': float(mcnemar_result.statistic),
            'pvalue': float(mcnemar_result.pvalue),
            'contingency_table': {
                'both_correct': int(contingency_table[0, 0]),
                'default_correct_custom_wrong': int(contingency_table[0, 1]),
                'default_wrong_custom_correct': int(contingency_table[1, 0]),
                'both_wrong': int(contingency_table[1, 1])
            },
            'significant': mcnemar_result.pvalue < 0.05,
            'better_model': 'default_head' if default_results['accuracy'] > custom_results['accuracy'] else 'custom_head'
        }
    }
    
    results_file = os.path.join(experiment_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Evaluation complete - results saved to {results_file}")

