import html
import re
import requests

from hydra.utils import instantiate

from src.datasets.collate import collate_fn_train, collate_fn_val
from src.utils.init_utils import set_worker_seed


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_datasets(config):
    """
    Create datasets for each of the dataset partitions.

    Args:
        config (DictConfig): hydra experiment config.

    Returns:
        datasets (dict[Dataset]): dict containing dataset instances for each partition.
    """
    datasets = {}
    for dataset_partition in config.datasets.keys():
        dataset_cfg = config.datasets[dataset_partition]

        # dataset partition init
        dataset = instantiate(dataset_cfg)  # instance transforms are defined inside

        datasets[dataset_partition] = dataset

    return datasets


def get_dataloaders(config, datasets, tokenization=None):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        datasets (dict[Dataset]): dict containing dataset instances for each partition.
        tokenization (Callable | None): tokenization function to be applied on the batch.
            If not None, it is passed to the dataloader collate_fn to be applied on the batch.
            This is required for text data, where tokenization is needed. For other data modalities, tokenization is not needed, so tokenization can be set to None.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        datasets (dict[Dataset]): dict containing dataset instances for each partition.
    """
    # dataloaders init
    dataloaders = {}
    for dataset_partition, dataset in datasets.items():
        dataloader_cfg = config.dataloaders[dataset_partition]

        assert dataloader_cfg.batch_size <= len(dataset), (
            f"The batch size ({dataloader_cfg.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        def get_collate_fn(split): # func to avoid contex binding
            collate_fn = collate_fn_train if split == "train" else collate_fn_val
            if tokenization is None:
                return collate_fn
            return lambda batch: collate_fn(tokenization(batch, split=split))

        partition_dataloader = instantiate(
            dataloader_cfg,
            dataset=dataset,
            collate_fn=get_collate_fn(dataset_partition),
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"), # No random access for IterableDatasets
            worker_init_fn=set_worker_seed,
        )

        dataloaders[dataset_partition] = partition_dataloader
 
    print(dataloaders["train"].collate_fn)
    return dataloaders


#################################
# Additional utils for datasets #
#################################

def download_file(url: str, path: str) -> bool:
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        path (str): The path where the downloaded file will be saved.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return False
    with open(path, 'wb') as f:
        f.write(response.content)
    return True


def clean_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing HTML tags, special characters, and extra spaces.

    Args:
        raw_text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = html.unescape(raw_text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text
