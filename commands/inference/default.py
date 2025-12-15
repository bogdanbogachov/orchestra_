from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from config import CONFIG

def run_infer_default(input_text, labels=None, adapter_path=None):
    model_config = CONFIG['model']
    model_path = CONFIG['paths']['model']
    
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

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

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
