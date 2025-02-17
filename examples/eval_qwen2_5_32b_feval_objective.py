from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.feval.feval_objective_gen_1 import feval_datasets as datasets
    from opencompass.configs.models.qwen.api_alicloud_qwen2_5_32b_chat import models
