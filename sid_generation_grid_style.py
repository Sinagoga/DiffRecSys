import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader, IterableDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingDataset(IterableDataset):
    
    def __init__(self, embeddings_path: str, batch_size: int = 256):
        self.embeddings_path = embeddings_path
        self.batch_size = batch_size
        
        if embeddings_path.endswith('.pkl') or embeddings_path.endswith('.pickle'):
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.item_ids = list(data.keys())
                    self.embeddings = np.array([data[iid] for iid in self.item_ids])
                else:
                    self.embeddings = data
                    self.item_ids = list(range(len(data)))
        elif embeddings_path.endswith('.npy'):
            self.embeddings = np.load(embeddings_path)
            self.item_ids = list(range(len(self.embeddings)))
        else:
            raise ValueError(f"Unsupported format: {embeddings_path}")
        
        logger.info(f"Loaded {len(self.embeddings)} embeddings of shape {self.embeddings[0].shape}")
    
    def __iter__(self):
        for i in range(0, len(self.embeddings), self.batch_size):
            batch_emb = self.embeddings[i:i + self.batch_size]
            batch_ids = self.item_ids[i:i + self.batch_size]
            yield torch.from_numpy(batch_emb).float(), batch_ids
    
    def get_all(self) -> Tuple[torch.Tensor, List]:
        embeddings_tensor = torch.from_numpy(self.embeddings).float()
        return embeddings_tensor, self.item_ids


class SIDGeneratorGRIDStyle:
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = None
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        logger.info(f"Checkpoint keys: {list(state_dict.keys())[:5]}...")
        
        # TODO:
        # self.model = ResidualQuantization(...)
        # self.model.load_state_dict(state_dict)
        # self.model.eval()

        self.state_dict = state_dict
        logger.info("Model checkpoint loaded successfully")
    
    def encode_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of embeddings into hierarchical SIDs.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        
        Returns:
            SIDs of shape (batch_size, num_hierarchies) with values in [0, vocab_size)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        embeddings = embeddings.to(self.device)
        
        with torch.no_grad():
            cluster_ids, _, _, _ = self.model(embeddings)
        
        return cluster_ids.cpu()
    
    def encode_all(
        self,
        embeddings: torch.Tensor,
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> torch.Tensor:
        all_sids = []
        n_items = len(embeddings)
        
        iterator = range(0, n_items, batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding embeddings to SIDs")
            except ImportError:
                pass
        
        for i in iterator:
            batch = embeddings[i:i + batch_size]
            batch_sids = self.encode_batch(batch)
            all_sids.append(batch_sids)
        
        return torch.cat(all_sids, dim=0)


class SIDToMaskGRFormat:
    def sids_to_dict(
        item_ids: List,
        sids: torch.Tensor,
    ) -> Dict[int, tuple]:
        sid_dict = {}
        
        for item_id, sid_row in zip(item_ids, sids):
            sid_tuple = tuple(sid_row.tolist())
            sid_dict[item_id] = sid_tuple
        
        return sid_dict
    
    def save_sids(
        sid_dict: Dict[int, tuple],
        output_path: str,
        format: str = 'pickle',
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(sid_dict, f)
            logger.info(f"Saved SIDs to {output_path} (pickle format)")
        
        elif format == 'json':
            json_dict = {str(k): v for k, v in sid_dict.items()}
            with open(output_path, 'w') as f:
                json.dump(json_dict, f)
            logger.info(f"Saved SIDs to {output_path} (JSON format)")
        
        elif format == 'pt':
            torch.save(sid_dict, output_path)
            logger.info(f"Saved SIDs to {output_path} (PyTorch format)")
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def load_sids(
        path: str,
        format: str = 'pickle',
    ) -> Dict[int, tuple]:
        if format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
                return {int(k): tuple(v) for k, v in data.items()}
        elif format == 'pt':
            return torch.load(path)
        else:
            raise ValueError(f"Unknown format: {format}")


def generate_sids_from_embeddings(
    embeddings_path: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 512,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_format: str = 'pickle',
) -> Dict[int, tuple]:
    logger.info("=" * 80)
    logger.info("GRID-style SID Generation Pipeline")
    logger.info("=" * 80)
    
    logger.info("\n[Step 1/4] Loading embeddings...")
    dataset = EmbeddingDataset(embeddings_path, batch_size=batch_size)
    embeddings, item_ids = dataset.get_all()
    logger.info(f"Loaded {len(item_ids)} embeddings")
    
    logger.info("\n[Step 2/4] Loading trained ResidualQuantization model...")
    generator = SIDGeneratorGRIDStyle(checkpoint_path, device=device)
    
    logger.info("\n[Step 3/4] Encoding embeddings to hierarchical SIDs...")
    sids = generator.encode_all(embeddings, batch_size=batch_size, show_progress=True)
    logger.info(f"Generated SIDs shape: {sids.shape}")
    logger.info(f"SID ranges: min={sids.min().item()}, max={sids.max().item()}")
    
    logger.info("\n[Step 4/4] Converting to MaskGR format and saving...")
    sid_dict = SIDToMaskGRFormat.sids_to_dict(item_ids, sids)
    SIDToMaskGRFormat.save_sids(sid_dict, output_path, format=output_format)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"SID generation complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total SIDs: {len(sid_dict)}")
    logger.info("=" * 80)
    
    return sid_dict

class SimpleHierarchicalQuantizer:
    
    def __init__(
        self,
        centroids_list: List[torch.Tensor],
        normalize_residuals: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.centroids_list = [c.to(device) for c in centroids_list]
        self.n_hierarchies = len(centroids_list)
        self.normalize_residuals = normalize_residuals
        self.device = torch.device(device)
        
        logger.info(f"Initialized quantizer with {self.n_hierarchies} hierarchies")
    
    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.to(self.device)
        sids = []
        residuals = embeddings
        
        for level, centroids in enumerate(self.centroids_list):
            if self.normalize_residuals and level > 0:
                residuals = torch.nn.functional.normalize(residuals, dim=-1)
            
            distances = torch.cdist(residuals, centroids)
            nearest_ids = torch.argmin(distances, dim=1)
            sids.append(nearest_ids)
            
            quantized = centroids[nearest_ids]
            residuals = residuals - quantized
        
        return torch.stack(sids, dim=1)
    
    @classmethod
    def from_sklearn_kmeans(
        cls,
        kmeans_models: List,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        centroids_list = [
            torch.from_numpy(km.cluster_centers_).float()
            for km in kmeans_models
        ]
        return cls(centroids_list, device=device)


def generate_sids_grid_style(
    embeddings_path: str,
    checkpoint_path: str,
    output_dir: str = './sids_output',
    dataset_name: str = 'beauty',
    num_hierarchies: int = 4,
    vocab_size: int = 256,
) -> str:
    output_path = os.path.join(output_dir, f'{dataset_name}_sids.pkl')
    
    sid_dict = generate_sids_from_embeddings(
        embeddings_path=embeddings_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        batch_size=512,
        output_format='pickle',
    )
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate SIDs using GRID's Residual Quantization approach"
    )
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings file (pkl or npy)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained RQ model checkpoint')
    parser.add_argument('--output', type=str, default='sids_output.pkl',
                       help='Path to save SIDs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--format', type=str, default='pickle',
                       choices=['pickle', 'json', 'pt'],
                       help='Output format')
    
    args = parser.parse_args()
    
    generate_sids_from_embeddings(
        embeddings_path=args.embeddings,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        output_format=args.format,
    )
