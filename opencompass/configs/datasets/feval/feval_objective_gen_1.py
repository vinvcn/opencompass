
from opencompass.datasets.feval import FEvalDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, SlidingWindowRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import last_option_postprocess_regex




feval_subject_mapping = {
    "业务专精能力":"业务专精能力",
    "中文通用基础能力":"中文通用基础能力",
    "金融能力":"金融能力"
}

feval_datasets = []
for _name in feval_subject_mapping.keys():
    _ch_name = feval_subject_mapping[_name][0]
    feval_infer_cfg=dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='user',
                        prompt=
                        f'以下是关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\n答案：'
                    ),
                    dict(role='assistant', prompt='答案是：{answer}'),
                ]),
                ice_token='</E>'
            ),
            # retriever=dict(type=SlidingWindowRetriever, k=5),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer)
        )
    
    feval_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=last_option_postprocess_regex, options='ABCD'))

    
    feval_datasets.append(
        dict(
            type=FEvalDataset,
            path='BYJK/feval_objective',
            # limit=1,
            name=_name,
            abbr='BYJK-FEval-客观题-' + _name, 
            reader_cfg=dict(
                input_columns=['question'],
                output_column='answer'),
            infer_cfg=feval_infer_cfg,
            eval_cfg=feval_eval_cfg,
        ))