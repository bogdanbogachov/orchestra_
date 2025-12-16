from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import os
from config import CONFIG
from logging import logger

def run_infer_default(input_text, labels=None, adapter_path=None):
    model_config = CONFIG['model']
    model_path = CONFIG['paths']['model']
    paths_config = CONFIG['paths']
    experiment_name = CONFIG.get('experiment', 'orchestra')
    
    if adapter_path is None:
        adapter_path = os.path.join(paths_config['experiments'], experiment_name, "default_head")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_config.get('pad_token', tokenizer.eos_token)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config['torch_dtype'], torch.float32)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=model_config['num_labels'],
        dtype=torch_dtype,
        device_map=model_config['device_map']
    )

    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"✓ Loaded finetuned score layer (classification head) from adapter")
    else:
        logger.info(f"Adapter path {adapter_path} not found, using base model with untrained score layer")

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    model_output_logits = outputs.logits
    probs = torch.nn.functional.softmax(model_output_logits, dim=-1)

    output = {
        'logits': model_output_logits,
        'probs': probs,
        'loss': outputs.loss
    }

    return output
