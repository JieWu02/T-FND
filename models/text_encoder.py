from torch import nn
from transformers import BertModel, BertConfig


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = self._get_bert_model(config)
        for p in self.model.parameters():
            p.requires_grad = self.config.trainable
        self.target_token_idx = 0
        self.text_encoder_embedding = dict()

    def forward(self, ids, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state[:, self.target_token_idx, :]
        return last_hidden_state

    def _get_bert_model(self, config):
        """Get BERT Chinese model"""
        if config.pretrained:
            return BertModel.from_pretrained(
                config.text_encoder_model, 
                output_attentions=False,
                output_hidden_states=True, 
                return_dict=True
            )
        else:
            return BertModel(config=BertConfig())
