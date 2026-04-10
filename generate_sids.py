import os
import torch
import numpy as np
import faiss
import glob
import tensorflow as tf
from omegaconf import DictConfig
import hydra
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def load_embeddings_from_tfrecords(items_dir):
    files = glob.glob(os.path.join(items_dir, "*.tfrecord.gz"))
    item2embs = defaultdict(np.ndarray)
    
    for file in files:
        raw_dataset = tf.data.TFRecordDataset(file, compression_type='GZIP')
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            f = example.features.feature
            item_id = f['id'].int64_list.value[0]
            embedding = np.array(f['embedding'].float_list.value)
            item2embs[item_id] = embedding

    return item2embs

def run_residual_quantization(embeddings, num_levels, codebook_size):
    num_items, dim = embeddings.shape
    sids = np.zeros((num_items, num_levels), dtype=np.int32)
    residuals = embeddings.copy()

    for level in range(num_levels):
        logger.info(f"Clustering level {level+1}...")
        kmeans = faiss.Kmeans(dim, codebook_size, niter=20, verbose=False)
        kmeans.train(residuals)
        
        _, indices = kmeans.index.search(residuals, 1)
        indices = indices.flatten()
        
        sids[:, level] = indices
        
        centroids = kmeans.centroids
        residuals -= centroids[indices]
        
    return sids

@hydra.main(version_base=None, config_path="./config", config_name="sid")
def main(cfg: DictConfig):
    logger.info(f"Loading data from {cfg.data.items_dir}")
    item2embs = load_embeddings_from_tfrecords(cfg.data.items_dir)
    item_ids = list(item2embs.keys())
    embeddings = np.array(list(item2embs.values()))
    logger.info(f"Loaded {len(item_ids)} items")

    sids = run_residual_quantization(
        embeddings, 
        num_levels=cfg.quantization.num_levels, 
        codebook_size=cfg.quantization.codebook_size
    )

    sid_dict = {item_id: sids[i].tolist() for i, item_id in enumerate(item_ids)}
    
    os.makedirs(os.path.dirname(cfg.data.output_sid_path), exist_ok=True)
    torch.save(sid_dict, cfg.data.output_sid_path)
    logger.info(f"SIDs saved to {cfg.data.output_sid_path}")

if __name__ == "__main__":
    main()