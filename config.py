from dataclasses import dataclass
from typing import Optional
@dataclass
class Config:
    data_dir: str = "./data/amazon_data/beauty/"
    items_path: str = "./data/amazon_data/beauty/items/"
    sids_output_path: str = "./data/amazon_data/beauty/sids/item_sids.pt"
    embedding_model: str = "google/flan-t5-small"
    batch_size_embed: int = 128
    num_clusters: int = 256
    num_hierarchies: int = 4
    vocab_size_per_hierarchy: int = 256
    normalize_residuals: bool = True
    use_rq_checkpoint: bool = False
    rq_checkpoint_path: Optional[str] = None
    num_layers: int = 4
    batch_size_train: int = 64
    learning_rate: float = 1e-4
    epochs: int = 1
    hidden_dim: int = 256
    device: str = "cpu"