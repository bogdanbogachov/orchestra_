import numpy as np
from sklearn.metrics import accuracy_score
from transformers import Trainer, DataCollatorWithPadding

from logger_config import logger


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def build_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset,
    val_dataset,
    callbacks,
):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    logger.info("✓ Trainer created")
    return trainer
