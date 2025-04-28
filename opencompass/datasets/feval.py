import csv
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from itertools import islice
from .base import BaseDataset

@LOAD_DATASET.register_module()
class FEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, limit: int=0, **kwargs):
        path = get_data_path(path)


        raw_data = []
        filename = osp.join(path, f'{name}.csv')
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader)  # skip the header
            if limit > 0:
                reader = islice(reader, limit)
            for row in reader:
                assert len(row) == 2
                raw_data.append({
                    'question': row[0],
                    'answer': row[1],
                })
        # dataset = DatasetDict({
        #     'test': Dataset.from_list(raw_data)
        # })
        return Dataset.from_list(raw_data)
