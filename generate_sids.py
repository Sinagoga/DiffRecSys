import torch
import json
from transformers import T5EncoderModel, AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
from config import Config
import os
import glob
import tensorflow as tf
from pathlib import Path
import pickle
import logging


try:
    from sid_generation_grid_style import (
        generate_sids_grid_style,
        SIDGeneratorGRIDStyle,
        SIDToMaskGRFormat,
        SimpleHierarchicalQuantizer,
        EmbeddingDataset,
    )
    GRID_STYLE_AVAILABLE = True
except ImportError:
    GRID_STYLE_AVAILABLE = False
    print("⚠️  sid_generation_grid_style module not found. Only basic KMeans clustering available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_texts_from_tfrecords(data_dir):
    file_pattern = os.path.join(data_dir, "items", "*.tfrecord.gz")
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"Файлы не найдены в {file_pattern}")

    texts_data = []
    
    raw_dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    
    print(f"Читаем данные из {len(files)} файлов...")
    
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        f = example.features.feature
        
        item_id = f['item_id'].bytes_list.value[0].decode('utf-8')
        
        item_text = f['text'].bytes_list.value[0].decode('utf-8')
        
        texts_data.append({
            'item_id': item_id,
            'text': item_text
        })
        
    return texts_data

def generate_embeddings(texts, cfg: Config):
    print(f"Загрузка модели {cfg.embedding_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model, use_fast=True)
    model = T5EncoderModel.from_pretrained(cfg.embedding_model).to('cuda')
    model.eval()

    embeddings = []
    item_ids = []
    
    print("Генерация эмбеддингов...")
    with torch.no_grad():
        for i in range(0, len(texts), cfg.batch_size_embed):
            batch = texts[i : i + cfg.batch_size_embed]
            batch_texts = [item['text'] for item in batch]
            batch_ids = [item['item_id'] for item in batch]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings.append(batch_emb)
            item_ids.extend(batch_ids)
            
    return item_ids, np.vstack(embeddings)

def load_embeddings(data_dir):
    file_pattern = os.path.join(data_dir, "items", "*.tfrecord.gz")
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"Файлы не найдены в {file_pattern}")
    
    raw_dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    
    print(f"Читаем данные из {len(files)} файлов...")
    
    embeddings = []
    item_ids = []

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        f = example.features.feature
        
        item_id = f['id'].int64_list.value[0]
        item_emb = f['embedding'].float_list.value
        item_ids.append(item_id)
        embeddings.append(item_emb)
        
    return item_ids, np.array(embeddings)

def generate_sids(embeddings, cfg: Config):
    print(f"Кластеризация (KMeans, кластеров: {cfg.num_clusters})...")
    # TODO: заменить на RQ-KMeans для лучшего качества SIDs (GRID-style).
    kmeans = KMeans(n_clusters=cfg.num_clusters, random_state=42, n_init="auto")
    sids = kmeans.fit_predict(embeddings)
    return sids

def generate_sids_hierarchical_from_checkpoint(
    embeddings: np.ndarray,
    checkpoint_path: str,
    num_hierarchies: int = 4,
    vocab_size: int = 256,
    batch_size: int = 512,
    device: str = 'cuda',
) -> np.ndarray:
    if not GRID_STYLE_AVAILABLE:
        raise ImportError(
            "sid_generation_grid_style module required. "
            "Make sure sid_generation_grid_style.py is in the same directory."
        )
    
    logger.info("=" * 80)
    logger.info("Generating Hierarchical SIDs (GRID-style)")
    logger.info("=" * 80)
    
    embeddings_tensor = torch.from_numpy(embeddings).float()
    
    logger.info(f"\n[1/3] Loading ResidualQuantization from {checkpoint_path}")
    try:
        generator = SIDGeneratorGRIDStyle(checkpoint_path, device=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    logger.info(f"\n[2/3] Encoding {len(embeddings)} embeddings to hierarchical SIDs")
    sids_tensor = generator.encode_all(
        embeddings_tensor,
        batch_size=batch_size,
        show_progress=True,
    )
    
    logger.info(f"\n[3/3] Verifying SID ranges")
    sids_np = sids_tensor.cpu().numpy()
    logger.info(f"SID shape: {sids_np.shape}")
    logger.info(f"SID value ranges per hierarchy:")
    for h in range(sids_np.shape[1]):
        min_val = sids_np[:, h].min()
        max_val = sids_np[:, h].max()
        logger.info(f"  Hierarchy {h}: [{min_val}, {max_val}]")
    
    logger.info("\n" + "=" * 80)
    logger.info("Hierarchical SID generation complete!")
    logger.info("=" * 80)
    
    return sids_np


def generate_sids_hierarchical_kmeans(
    embeddings: np.ndarray,
    num_hierarchies: int = 4,
    vocab_size: int = 256,
    normalize_residuals: bool = True,
) -> np.ndarray:
    logger.info("=" * 80)
    logger.info("Generating Hierarchical SIDs (Sequential KMeans)")
    logger.info("=" * 80)
    logger.info(f"Input embeddings shape: {embeddings.shape}")
    logger.info(f"Number of hierarchies: {num_hierarchies}")
    logger.info(f"Vocabulary size per level: {vocab_size}")
    
    all_sids = []
    current_residuals = embeddings.copy()
    
    for hierarchy_level in range(num_hierarchies):
        logger.info(f"\n[{hierarchy_level + 1}/{num_hierarchies}] Training KMeans for hierarchy level {hierarchy_level}")
        
        if normalize_residuals and hierarchy_level > 0:
            residuals_norm = np.linalg.norm(current_residuals, axis=1, keepdims=True)
            current_residuals = current_residuals / (residuals_norm + 1e-8)
            logger.info(f"  Residuals normalized")
        
        logger.info(f"  Training KMeans with {vocab_size} clusters...")
        kmeans = KMeans(
            n_clusters=vocab_size,
            random_state=42,
            n_init=10,
            verbose=0,
        )
        level_sids = kmeans.fit_predict(current_residuals)
        all_sids.append(level_sids)
        
        centroids = kmeans.cluster_centers_
        quantized = centroids[level_sids]
        current_residuals = current_residuals - quantized
        
        residual_norm = np.linalg.norm(current_residuals)
        logger.info(f"Residual L2 norm: {residual_norm:.4f}")
        logger.info(f"Unique clusters assigned: {len(np.unique(level_sids))}")
    
    sids = np.stack(all_sids, axis=1)
    
    logger.info("\n" + "=" * 80)
    logger.info("Hierarchical SID generation complete!")
    logger.info(f"Output shape: {sids.shape}")
    logger.info(f"Output dtype: {sids.dtype}")
    logger.info("=" * 80)
    
    return sids


def sids_to_dict(
    item_ids: list,
    sids: np.ndarray,
) -> dict:
    sid_dict = {}
    for item_id, sid_row in zip(item_ids, sids):
        sid_tuple = tuple(sid_row.tolist())
        sid_dict[item_id] = sid_tuple
    
    return sid_dict


def save_sids(
    sid_dict: dict,
    output_path: str,
    format: str = 'pickle',
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(sid_dict, f)
        logger.info(f"✓ Saved SIDs to {output_path} (pickle format)")
    
    elif format == 'json':
        json_dict = {str(k): v for k, v in sid_dict.items()}
        with open(output_path, 'w') as f:
            json.dump(json_dict, f)
        logger.info(f"✓ Saved SIDs to {output_path} (JSON format)")
    
    elif format == 'pt':
        torch.save(sid_dict, output_path)
        logger.info(f"✓ Saved SIDs to {output_path} (PyTorch format)")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_sids(path: str, format: str = 'pickle') -> dict:
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

def main_grid_style_hierarchical(cfg: Config):
    logger.info("\n" + "=" * 80)
    logger.info("HIERARCHICAL SID GENERATION (GRID-STYLE)")
    logger.info("=" * 80)
    
    logger.info("\n[Step 1/3] Loading embeddings from items/...")
    try:
        item_ids, embeddings = load_embeddings(cfg.data_dir)
        logger.info(f"✓ Loaded {len(item_ids)} embeddings with shape {embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return
    
    logger.info(f"\n[Step 2/3] Generating {cfg.num_hierarchies}-level hierarchical SIDs...")
    
    if cfg.use_rq_checkpoint and cfg.rq_checkpoint_path:
        logger.info(f"Using trained RQ checkpoint: {cfg.rq_checkpoint_path}")
        sids = generate_sids_hierarchical_from_checkpoint(
            embeddings=embeddings,
            checkpoint_path=cfg.rq_checkpoint_path,
            num_hierarchies=cfg.num_hierarchies,
            vocab_size=cfg.vocab_size_per_hierarchy,
            batch_size=cfg.batch_size_embed,
            device=cfg.device,
        )
    else:
        logger.info("Training hierarchical KMeans (no checkpoint provided)")
        sids = generate_sids_hierarchical_kmeans(
            embeddings=embeddings,
            num_hierarchies=cfg.num_hierarchies,
            vocab_size=cfg.vocab_size_per_hierarchy,
            normalize_residuals=cfg.normalize_residuals,
        )
    
    logger.info(f"\n[Step 3/3] Saving SIDs...")
    sid_dict = sids_to_dict(item_ids, sids)
    
    output_dir = Path(cfg.sids_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_sids(sid_dict, cfg.sids_output_path, format='pickle')
    logger.info(f"✓ Saved {len(sid_dict)} SIDs to {cfg.sids_output_path}")
    
    return sid_dict


def main_simple_kmeans(cfg: Config):
    logger.info("\n" + "=" * 80)
    logger.info("SIMPLE SID GENERATION (KMeans)")
    logger.info("=" * 80)
    
    logger.info("\n[Step 1/3] Loading embeddings...")
    try:
        item_ids, embeddings = load_embeddings(cfg.data_dir)
        logger.info(f"✓ Loaded {len(item_ids)} embeddings")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return
    
    logger.info(f"\n[Step 2/3] Running KMeans with {cfg.num_clusters} clusters...")
    sids = generate_sids(embeddings, cfg)
    
    logger.info(f"\n[Step 3/3] Saving SIDs...")
    sid_dict = {item_id: (sid,) for item_id, sid in zip(item_ids, sids)}
    
    output_dir = Path(cfg.sids_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_sids(sid_dict, cfg.sids_output_path, format='pickle')
    logger.info(f"✓ Saved {len(sid_dict)} SIDs to {cfg.sids_output_path}")
    
    return sid_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SIDs for DiffRecSys")
    parser.add_argument(
        '--method',
        type=str,
        default='hierarchical',
        choices=['hierarchical', 'simple', 'checkpoint'],
        help='Method for SID generation'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to RQ checkpoint (for checkpoint method)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to data directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    cfg = Config()
    
    if args.data_dir:
        cfg.data_dir = args.data_dir
    
    if args.checkpoint:
        cfg.use_rq_checkpoint = True
        cfg.rq_checkpoint_path = args.checkpoint
    
    if args.method == 'hierarchical':
        logger.info("Selected: Hierarchical SID generation")
        main_grid_style_hierarchical(cfg)
    
    elif args.method == 'simple':
        logger.info("Selected: Simple KMeans SID generation")
        main_simple_kmeans(cfg)
    
    elif args.method == 'checkpoint':
        if not cfg.rq_checkpoint_path:
            logger.error("--checkpoint required for checkpoint method")
            exit(1)
        logger.info("Selected: Checkpoint-based SID generation")
        cfg.use_rq_checkpoint = True
        main_grid_style_hierarchical(cfg)
