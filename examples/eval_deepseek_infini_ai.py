from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets as datasets
    from opencompass.configs.models.qwen.hf_qwen2_0_5b_instruct_cpu import models
