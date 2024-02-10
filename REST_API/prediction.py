from transformers import AutoTokenizer
from model import HateDetector, tokenizer_name, device
import torch

_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
_model = HateDetector().eval()
_model.load_state_dict(torch.load("out.pt"))


def predict(text: str):
    with torch.no_grad():
        tokens = _tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        y_pred = _model(tokens.input_ids.to(device), tokens.attention_mask.to(device))
        y_pred = y_pred[0]

    if y_pred[0] < 0.5:
        return "This is not a Hateful post"
    else:
        return "This is a Hateful text"
