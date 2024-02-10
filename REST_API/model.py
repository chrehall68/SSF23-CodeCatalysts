import torch
from transformers import AutoModelForSequenceClassification

device = "cpu"
tokenizer_name = "bert-base-uncased"
model_name = "chreh/bert-discrimination-classifier"


class HateDetector(torch.nn.Module):
    def __init__(self):
        super(HateDetector, self).__init__()
        self.text_model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            .to(device, dtype=torch.float32)
            .train()
        )
        self.output_func = torch.nn.Softmax(dim=-1)

    def forward(self, tokens: torch.Tensor, token_attention_mask: torch.Tensor):
        return self.output_func(
            self.text_model(tokens, attention_mask=token_attention_mask).logits
        )
