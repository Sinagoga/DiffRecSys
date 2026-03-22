import logging
from typing import Optional
from pathlib import Path
import os
import gzip
import json
from collections import defaultdict

from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import download_file, clean_text


logger = logging.getLogger(__name__)


AMAZON_REVIEW_DATASET_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{section}_{category}{'_5' if section == 'reviews' else ''}.json.gz"
AMAZON_REVIEW_CATEGORIES = [
    'Books',
    'Electronics',
    'Movies_and_TV',
    'CDs_and_Vinyl',
    'Clothing_Shoes_and_Jewelry',
    'Home_and_Kitchen',
    'Kindle_Store',
    'Sports_and_Outdoors',
    'Cell_Phones_and_Accessories',
    'Health_and_Personal_Care',
    'Toys_and_Games',
    'Video_Games',
    'Tools_and_Home_Improvement',
    'Beauty',
    'Apps_for_Android',
    'Office_Products',
    'Pet_Supplies',
    'Automotive',
    'Grocery_and_Gourmet_Food',
    'Patio_Lawn_and_Garden',
    'Baby',
    'Digital_Music',
    'Musical_Instruments',
    'Amazon_Instant_Video'
]


class AmazonReviewDataset(BaseDataset):
    def __init__(
        self,
        category: str,
        part: str = "train",
        cache_dir: Path = "./cache/AmazonReviews2014/",
        limit: Optional[int] = None,

        leave_one_out: bool = True,
        max_history_length: Optional[int] = None,
        min_history_length: Optional[int] = None,
    ):
        """
        Args:
            part: Dataset split - "train", "val" or "test".
            category: Dataset category, one of AMAZON_REVIEW_CATEGORIES.
            cache_dir: Directory to store the raw and processed data.
            limit: If not None, limit the total number of elements.
        """
        assert part in ["train", "val"], (
            f"Part '{part}' is not recognized."
        )
        assert category in AMAZON_REVIEW_CATEGORIES, (
            f"Category '{category}' is not is not a part of {AMAZON_REVIEW_CATEGORIES}."
        )

        self.part = part
        self.category = category
        self.cache_dir = cache_dir / f"{category}_{part}"
        
        self.raw_data_dir = self.cache_dir / 'raw'
        self.processed_data_dir = self.cache_dir / 'processed'

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        all_item_seqs, self.id_mapping, self.item2meta = self._download_and_process_raw()
        index = self._prepare_split(all_item_seqs, split=part, leave_one_out=leave_one_out, max_history_length=max_history_length, min_history_length=min_history_length)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=part == "train",
            # instance_transforms=None,
        )

    def _download_raw(self, path: Path, category: str, section: str = 'reviews') -> Path:
        url = AMAZON_REVIEW_DATASET_URL.format(
            section=section,
            category=category
        )
        local_filepath = path / Path(url).name
        if not os.path.exists(local_filepath):
            download_file(url, local_filepath)
        return local_filepath

    def _process_reviews(self,
        input_path: Path,
        output_path: Path
    ) -> tuple[dict, dict]:
        """
        Process the reviews from the input path and save the data to the output path.

        Args:
            input_path (Path): The path to the input file containing the reviews.
            output_path (Path): The path to save the data.

        Returns:
            all_item_seqs (dict): A dictionary containing the user-item sequences.
            id_mapping (dict): A dictionary containing data maps.
        """

        def load_reviews(path: Path) -> tuple[dict, dict]:
            item_seqs = defaultdict(list)
            
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc='Loading reviews'):
                    inter = json.loads(line)

                    user = inter['reviewerID']
                    item = inter['asin']
                    time = inter['unixReviewTime']
                    
                    item_seqs[user].append((item, time))

            user2id = {'[PAD]': 0}
            item2id = {'[PAD]': 0}
            id2user = ['[PAD]']
            id2item = ['[PAD]']

            item_seqs_processed = {}

            for user, item_time in tqdm(item_seqs.items(), desc='Remapping IDs'):
                user2id[user] = len(id2user)
                id2user.append(user)
                
                item_time.sort(key=lambda x: x[1])
                item_time = [item for item, _ in item_time]
                item_seqs_processed[user] = item_time
                
                for item in item_time:
                    if item not in item2id:
                        item2id[item] = len(id2item)
                        id2item.append(item)

            id_mapping = {
                'user2id': user2id,
                'item2id': item2id,
                'id2user': id2user,
                'id2item': id2item
            }

            return item_seqs_processed, id_mapping

        # Check if the processed data already exists
        seq_file = output_path / 'all_item_seqs.json'
        id_mapping_file = output_path / 'id_mapping.json'

        if seq_file.exists() and id_mapping_file.exists():
            logging.info('Restoring processed reviews from cache...')

            with open(seq_file, 'r') as f:
                item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
        else:
            logging.info('Processing reviews...')
            
            item_seqs, id_mapping = load_reviews(input_path)

            with open(seq_file, 'w') as f:
                json.dump(item_seqs, f)
            with open(id_mapping_file, 'w') as f:
                json.dump(id_mapping, f)
    
        return item_seqs, id_mapping

    def _process_meta(
        self,
        input_path: Path,
        output_path: Path
    ) -> Optional[dict]:
        """
        Process metadata based on the specified process type.

        Args:
            input_path (str): The path to the input metadata file.
            output_path (str): The path to save the processed metadata file.

        Returns:
            dict: A dictionary containing the item metadata.

        Raises:
            NotImplementedError: If the metadata processing type is not implemented.
        """

        def load_metadata(
            path: Path,
            item_asins: set
        ) -> dict:
            data = {}
            
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc='Loading metadata'):
                    info = json.loads(line)
                    if info['asin'] in item_asins:
                        data[info['asin']] = info
            
            return data

        def extract_meta_sentences(
            metadata: dict
        ) -> dict:
            def sent_process(raw: Optional[str | float | list]) -> str:
                if raw is None:
                    return ''

                if isinstance(raw, float):
                    return str(raw)
                
                if isinstance(raw, list):
                    proccessed = [sent_process(v) for v in raw]
                    proccessed = [p for p in proccessed if p]
                    return ', '.join(proccessed)
                
                return clean_text(raw).strip().rstrip('.')

            features_mapping = {
                'title': 'Title',
                'price': 'Price',
                'brand': 'Brand',
                'categories': 'Category',
                'feature': 'Features',
                'description': 'Description'
            }
            
            item2meta = {}
            for item, meta in tqdm(metadata.items(), desc='Extracting meta sentences'):
                meta_parts = []
                
                for feature, label in features_mapping.items():
                    processed_value = sent_process(meta.get(feature))
                    if processed_value:
                        meta_parts.append(f"{label}: {processed_value}.")
                
                item2meta[item] = ' '.join(meta_parts)
            return item2meta

        process_mode = self.config['metadata']
        meta_file = output_path / f'metadata.{process_mode}.json'

        if process_mode == 'none':
            return None

        if meta_file.exists():
            logging.info('Restoring processed metadata from cache...')

            with open(meta_file, 'r') as f:
                item2meta = json.load(f)
        else:
            logging.info(f'Processing metadata, mode: {process_mode}')

            item2meta = load_metadata(
                path=input_path,
                item_asins=set(self.id_mapping['item2id'].keys())
            )
            if process_mode == 'raw':
                pass
            elif process_mode == 'sentence':
                item2meta = extract_meta_sentences(metadata=item2meta)
            else:
                raise NotImplementedError('Metadata processing type not implemented.')

            with open(meta_file, 'w') as f:
                json.dump(item2meta, f)
            
        return item2meta

    def _download_and_process_raw(self) -> tuple[dict, dict, Optional[dict]]:
        """
        Downloads and processes the raw data files.

        This method downloads the raw data files for reviews and metadata from the specified path,
        processes the raw data, and saves the processed data in the cache directory.

        Returns:
            None
        """

        reviews_localpath = self._download_raw(
            path=self.raw_data_dir,
            category=self.category,
            section='reviews'
        )
        meta_localpath = self._download_raw(
            path=self.raw_data_dir,
            category=self.category,
            section='meta'
        )


        all_item_seqs, id_mapping = self._process_reviews(
            input_path=reviews_localpath,
            output_path=self.processed_data_dir
        )
        item2meta = self._process_meta(
            input_path=meta_localpath,
            output_path=self.processed_data_dir
        )

        return all_item_seqs, id_mapping, item2meta
    
    def _prepare_split(
            self,
            all_item_seqs: dict,
            split: str = 'train',
            leave_one_out: bool = True,
            max_history_length: Optional[int] = None,
            min_history_length: Optional[int] = None
        ) -> list[dict]:
        """
        Prepare the dataset split by creating an index of data samples.

        Args:
            all_item_seqs (dict): Dictionary mapping user IDs to item sequences.
            split (str): The dataset split to prepare, either 'train', 'val', or 'test'.
            leave_one_out (bool): Whether to use the leave-one-out strategy for splitting.
        """

        index = []

        for user, item_seq in all_item_seqs.items():
            if split == 'test':
                index.append({
                    'user': user,
                    'history': item_seq[:-1],
                    'target': item_seq[-1],
                })
            elif split == 'val' and len(item_seq) > 1:
                index.append({
                    'user': user,
                    'history': item_seq[:-1],
                    'target': item_seq[-1],
                })
            elif split == 'train' and len(item_seq) > 2:
                if leave_one_out:
                    index.append({
                        'user': user,
                        'history': item_seq[:-2],
                        'target': item_seq[-2],
                    })
                else:
                    for i in range(min_history_length, len(item_seq) - 1):
                        if max_history_length and i > max_history_length:
                            break
                        index.append({
                            'user': user,
                            'history': item_seq[:i],
                            'target': item_seq[i],
                        })

        return index
