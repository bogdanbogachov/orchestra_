from datasets import load_dataset
import json

dataset = load_dataset("PolyAI/banking77")

with open("train_data.json", "w") as f:
    json.dump(dataset["train"].to_list(), f, indent=2)

with open("test_data.json", "w") as f:
    json.dump(dataset["test"].to_list(), f, indent=2)