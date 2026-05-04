import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OPENAI_API_KEY"] = "your-openai-key-here"

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import json

with open("data/eval_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)

result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)