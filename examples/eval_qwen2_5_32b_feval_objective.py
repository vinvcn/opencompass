from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from opencompass.configs.datasets.feval.feval_objective_gen_1 import feval_datasets as datasets
    from opencompass.configs.datasets.FinanceIQ.FinanceIQ_gen import financeIQ_datasets as datasets
    from bobfintech.configs.models.baidu import models


# for d in datasets:
#     d['reader_cfg']['test_range'] = '[0:4]'

# Local Runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=20, min_task_size=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=20,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask),
    ),
)
