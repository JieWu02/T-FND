from torch import nn
from transformers import ViTModel, ViTConfig


class ImageEncoder(nn.Module):
    """Vision Transformer (ViT) Image Encoder"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self._get_vit_model(config)
        
        for p in self.model.parameters():
            p.requires_grad = config.trainable

        self.target_token_idx = 0
        self.image_encoder_embedding = dict()

    def forward(self, ids, image):
        output = self.model(image)
        last_hidden_state = output.last_hidden_state[:, self.target_token_idx, :]
        return last_hidden_state

    def _get_vit_model(self, config):
        """Get Vision Transformer model"""
        if config.pretrained:
            import warnings
            import logging
            
            # 临时设置日志级别
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*Some weights of ViTModel were not initialized.*")
                return ViTModel.from_pretrained(
                    config.image_model_name, 
                    output_attentions=False,
                    output_hidden_states=True, 
                    return_dict=True
                )
        else:
            return ViTModel(config=ViTConfig())

