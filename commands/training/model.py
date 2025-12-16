import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from models.custom_llama_classification import LlamaClassificationHead
from peft import LoraConfig, get_peft_model, TaskType
from config import CONFIG
from logging import logger

class CustomClassificationModel(torch.nn.Module):
    def __init__(self, base_model, classifier):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.config = base_model.config
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        result = self.classifier(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels
        )

        return SequenceClassifierOutput(
            logits=result['logits'],
            loss=result['loss']
        )


def load_model_and_tokenizer():
    model_config = CONFIG['model']
    model_path = CONFIG['paths']['model']
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_config['pad_token']
    
    use_custom_head = model_config['use_custom_head']
    num_labels = model_config['num_labels']
    
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config['torch_dtype'], torch.float32)
    
    if use_custom_head:
        base_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=model_config['device_map']
        )
        
        classifier = LlamaClassificationHead(
            config=base_model.config,
            num_labels=num_labels,
            pooling_strategy=model_config['pooling_strategy'],
            use_fft=model_config['use_fft']
        ).to(base_model.device)
        
        model = CustomClassificationModel(base_model, classifier)
        logger.info(f"✓ Loaded base model with custom classification head")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch_dtype,
            device_map=model_config['device_map']
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        logger.info("✓ Loaded model with default classification head")

    return model, tokenizer, use_custom_head


def setup_lora(model, use_custom_head: bool):
    lora_config_dict = CONFIG['lora']
    
    if use_custom_head:
        module_name = "classifier"
        task_type = TaskType.FEATURE_EXTRACTION
        target_model = model.base_model
    else:
        module_name = "score"
        task_type = TaskType.SEQ_CLS
        target_model = model

    lora_config = LoraConfig(
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['lora_alpha'],
        target_modules=lora_config_dict['target_modules'],
        lora_dropout=lora_config_dict['lora_dropout'],
        bias=lora_config_dict['bias'],
        task_type=task_type,
        modules_to_save=[module_name]
    )

    if use_custom_head:
        model.base_model = get_peft_model(target_model, lora_config)
    else:
        model = get_peft_model(target_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ LoRA configured")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Classification head is trainable")

    return model
