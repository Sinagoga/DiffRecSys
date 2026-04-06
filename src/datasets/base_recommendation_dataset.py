import logging
from typing import Optional

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class BaseRecommendationDataset(BaseDataset):
    """
    Extended base dataset class, which includes additional attributes and methods for the recommendation datasets, such as user2id and item2id mappings, and methods to calculate the number of users, items, interactions, and average item sequence length.
    """

    def __init__(
        self,
        index: list,
        all_item_seqs: Optional[dict] = None,
        id_mapping: Optional[dict] = None,
        item2meta: Optional[dict] = None,
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms=None,
    ):
        """
        Args:
            all_item_seqs (dict): Dictionary mapping user IDs to item sequences.
            id_mapping (dict): Dictionary containing ID mappings for users and items.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """

        self.all_item_seqs = all_item_seqs or {}
        self.id_mapping = id_mapping or {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = item2meta or {}

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        if not self.user2id:
            return len(set([element.get('user', None) for element in self._index]))
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        if not self.item2id:
            return len(set.union(
                set(element.get('history', []) + [element.get('target', None)]) for element in self._index
            ))
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        if not self.all_item_seqs:
            return len(self._index)

        n_inters = 0
        for user, item_seq in self.all_item_seqs.items():
            n_inters += len(item_seq)
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping.get('user2id', {})

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping.get('item2id', {})
