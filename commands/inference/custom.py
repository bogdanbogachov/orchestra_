from transformers import AutoTokenizer, AutoModel
from models.custom_llama_classification import LlamaClassificationHead
from peft import PeftModel
import torch
import os
from config import CONFIG
from logger_config import logger

def run_infer_custom(input_text, labels=None, adapter_path=None):
    model_config = CONFIG['model']
    model_path = CONFIG['paths']['model']
    paths_config = CONFIG['paths']
    experiment_name = CONFIG.get('experiment', 'orchestra')
    
    if adapter_path is None:
        adapter_path = os.path.join(paths_config['experiments'], experiment_name, "custom_head")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_config.get('pad_token', tokenizer.eos_token)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config['torch_dtype'], torch.float32)
    
    base_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=model_config['device_map']
    )

    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        base_model = PeftModel.from_pretrained(base_model, adapter_path)

        classifier_path = os.path.join(adapter_path, "classifier.pt")
        if os.path.exists(classifier_path):
            logger.info(f"Loading fine-tuned classifier from {classifier_path}")
            classifier = torch.load(classifier_path, map_location=base_model.device)
        else:
            logger.info("Classifier not found, using randomly initialized classifier")
            classifier = LlamaClassificationHead(
                config=base_model.config,
                num_labels=model_config['num_labels'],
                pooling_strategy=model_config['pooling_strategy'],
                use_fft=model_config['use_fft']
            ).to(base_model.device)
    else:
        logger.info(f"Adapter path {adapter_path} not found.")

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state

    output = classifier(
        hidden_states=hidden_states,
        attention_mask=inputs.get("attention_mask"),
        labels=labels
    )
    
    return output
