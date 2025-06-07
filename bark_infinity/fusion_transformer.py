import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class FusionTransformer(nn.Module):
    """Simple mixture-of-experts transformer fusion.

    Each expert is an arbitrary Hugging Face transformer model. The module learns
    a weight for each expert and combines their hidden states.
    """

    def __init__(self, model_names):
        super().__init__()
        if not model_names:
            raise ValueError("model_names must contain at least one model identifier")
        self.tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names]
        self.models = nn.ModuleList([AutoModel.from_pretrained(name) for name in model_names])
        self.logits_weight = nn.Parameter(torch.ones(len(self.models)))

    def encode(self, text):
        """Tokenize input text using the first tokenizer."""
        tokenizer = self.tokenizers[0]
        return tokenizer(text, return_tensors="pt")

    def forward(self, input_ids, attention_mask=None):
        hidden_states = []
        for model in self.models:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states.append(outputs.last_hidden_state)
        stacked = torch.stack(hidden_states, dim=0)
        weights = F.softmax(self.logits_weight, dim=0)
        fused = torch.sum(weights.view(-1, 1, 1, 1) * stacked, dim=0)
        return fused
