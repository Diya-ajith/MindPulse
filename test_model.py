import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.nn.functional as F

MODEL_DIR = "bert_goemotions"

# load tokenizer & model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# test input
text = "I feel very nervous about my future"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# prediction
probs = F.softmax(outputs.logits, dim=1)
pred = torch.argmax(probs, dim=1).item()

print("Predicted label index:", pred)
print("Confidence:", round(probs[0][pred].item() * 100, 2), "%")


from labels import LABELS

emotion = LABELS[pred]
print("Emotion:", emotion)
