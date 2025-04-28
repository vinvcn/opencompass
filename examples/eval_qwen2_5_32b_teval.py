from mmengine.config import read_base
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from opencompass.configs.datasets.feval.feval_objective_gen_1 import feval_datasets, f_eval_summary_groups
    # from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # # from opencompass.configs.datasets.cmmlu.cmmlu_gen_c1365 import cmmlu_datasets
    # # from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups
    # from opencompass.configs.summarizers.groups.ceval import ceval_summary_groups
    from opencompass.configs.datasets.teval.teval_zh_gen import teval_datasets as datasets_zh
    from opencompass.configs.datasets.teval.teval_en_gen import teval_datasets as datasets_en
    from opencompass.configs.summarizers.teval import summarizer
    from opencompass.configs.models.qwen.api_alicloud_qwen2_5_32b_chat import models

print("import done " * 5)
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=20),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=20,
        concurrent_users=20,
        # retry=100,  # Modify if needed
        task=dict(type=OpenICLInferTask)),
)

datasets = datasets_zh + datasets_en 

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:19]'
    
    
# with read_base():
#     # from opencompass.configs.datasets.feval.feval_objective_gen_1 import feval_datasets as datasets
#     from opencompass.configs.datasets.teval.teval_zh_gen import teval_datasets as datasets_zh
#     from opencompass.configs.datasets.teval.teval_en_gen import teval_datasets as datasets_en
#     from opencompass.configs.summarizers.teval import summarizer
#     from opencompass.configs.models.qwen.api_alicloud_qwen2_5_32b_chat import models


# datasets = datasets_en + datasets_zh

# for d in datasets:
#     d['reader_cfg']['test_range'] = '[0:4]'



# Local Runner
# infer = dict(
#     partitioner=dict(type=NumWorkerPartitioner, num_worker=10, min_task_size=1),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=10,
#         retry=0,  # Modify if needed
#         task=dict(type=OpenICLInferTask),
#     ),
# )