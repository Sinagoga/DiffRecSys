from src.maskgr_modules.experimental.modules.rotary_position_encoding import RotaryTransformerEncoder, RotaryTransformerEncoderLayer
import torch.nn as nn

class MaskGRModel(nn.Module):
    def __init__(self, num_items, embedding_dim, nhead=8, num_layers=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        encoder_layer = RotaryTransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=0.25,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = RotaryTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(embedding_dim, num_items)

    def forward(self, item_seq, mask=None):
        x = self.item_emb(item_seq) 
        x = self.transformer(x, src_key_padding_mask=mask)
        logits = self.output_layer(x)
        return logits