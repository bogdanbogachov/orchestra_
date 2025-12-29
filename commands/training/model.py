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
from logger_config import logger

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
            input_ids=input_ids,  # Pass input_ids for pad_token_id logic in "last" pooling
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
    
    # ALWAYS load the default model first to get its score layer weights
    # This ensures we get the exact same initialization for both custom and default heads
    default_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        torch_dtype=torch_dtype,
        device_map=model_config['device_map']
    )
    
    if default_model.config.pad_token_id is None:
        default_model.config.pad_token_id = tokenizer.pad_token_id
    
    if use_custom_head:
        # Extract the base model from the default model (without classification head)
        base_model = default_model.model
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
        
        pooling_strategy = model_config['pooling_strategy']
        use_fft = model_config['use_fft']
        use_default_style = model_config.get('use_default_style', False)
        logger.info(f"Custom head configuration - Pooling type: {pooling_strategy}, FFT used: {use_fft}, Default style: {use_default_style}")
        
        classifier = LlamaClassificationHead(
            config=base_model.config,
            num_labels=num_labels,
            pooling_strategy=pooling_strategy,
            use_fft=use_fft,
            use_default_style=use_default_style
        )
        
        # Determine target device - handle device_map='auto' case
        if hasattr(base_model, 'device'):
            target_device = base_model.device
        elif hasattr(base_model, 'hf_device_map'):
            # For device_map='auto', get the device of the first parameter
            first_param = next(base_model.parameters())
            target_device = first_param.device
        else:
            # Fallback to CPU
            target_device = torch.device('cpu')
        
        # Move classifier to target device BEFORE copying weights
        classifier = classifier.to(target_device)
        
        # Copy weights from the actual default model's score layer
        # This ensures we use the exact same weights that the default head uses
        with torch.no_grad():
            # Get target dtype from classifier
            target_dtype = classifier.classifier.weight.dtype
            # Get source weight from the actual default model (same one that would be used for default head)
            source_weight = default_model.score.weight.data.to(device=target_device, dtype=target_dtype)
            # Copy the weight
            classifier.classifier.weight.data.copy_(source_weight)
        
        logger.info("✓ Initialized custom classifier weights to match default head's score layer")
        
        model = CustomClassificationModel(base_model, classifier)
        logger.info(f"✓ Loaded base model with custom classification head")
        
        # Clean up the default model (we only needed it for the weights and base model)
        del default_model
    else:
        # Use the default model directly
        model = default_model
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
    if use_custom_head:
        logger.info(f"  Custom classifier head is trainable (will be saved separately)")
    else:
        logger.info(f"  Default score layer (classification head) is trainable (will be saved in adapter)")

    return model
