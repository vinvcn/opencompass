from opencompass.datasets.subjective.feval import FEvalSubjectiveDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator

from opencompass.summarizers.subjective.feval_subjective import FEvalSubjectiveSummerizer


subject_mapping = {
    "业务专精能力":["业务专精能力"],
    "中文通用基础能力":["中文通用基础能力"],
    "金融能力":["金融能力"]
}

subjective_reader_cfg = dict(
    input_columns=['question', 'capability', 'critiquellm_prefix'],
    output_column='judge',
    )

data_path ='BYJK/feval_subjective'


feval_subjective_datasets = []

for _name in subject_mapping:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='user',
                        prompt='{question}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='user',
                        prompt = '{critiquellm_prefix}[助手的答案开始]\n{prediction}\n[助手的答案结束]\n'
                    ),
                ]),
            ),
        ),
        pred_role='assistant',
    )

    feval_subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=FEvalSubjectiveDataset,
            limit=1,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
            summarizer = dict(type=FEvalSubjectiveSummerizer, judge_type='general')
        ))
