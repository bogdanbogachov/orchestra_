from datasets import load_dataset
import json

dataset = load_dataset("PolyAI/banking77")

with open("banking77_train.json", "w") as f:
    json.dump(dataset["train"].to_list(), f, indent=2)

with open("banking77_test.json", "w") as f:
    json.dump(dataset["test"].to_list(), f, indent=2)


ds = load_dataset("clinc_oos", "plus")

def rename_key(split):
    return [{"text": x["text"], "label": x["intent"]} for x in split]

train = rename_key(ds["train"])
test  = rename_key(ds["test"])

with open("clinc150_train.json", "w") as f:
    json.dump(train, f, indent=2)

with open("clinc150_test.json", "w") as f:
    json.dump(test, f, indent=2)
