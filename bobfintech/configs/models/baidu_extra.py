
from opencompass.models import OpenAI
from opencompass.utils.text_postprocessors import general_cn_postprocess, first_capital_postprocess
API_KEY = 'bce-v3/ALTAK-vFe6Wkli8eUiYXFGhTRas/098bbb088294d66dbf0298cc4781847b5a27a1e1'

models = [
    dict(
        type=OpenAI,
        abbr='ERNIE-4.5-Turbo-32K',
        path='ernie-4.5-turbo-32k',
        key=API_KEY,
        openai_api_base='https://qianfan.baidubce.com/v2/chat/completions',
        max_completion_tokens=8192,
        batch_size=2,
        run_cfg=dict(num_gpus=0),
        pred_postprocessor={
            'BYKJ-FEval-主观题*': dict(type=general_cn_postprocess),
            # '*': dict(type=first_capital_postprocess)
            },
        # default_headers={"appid":"app-s8p6lY9N"}
    ),
    dict(
        type=OpenAI,
        abbr='ERNIE-X1-Turbo-32K',
        path='ernie-x1-turbo-32k',
        key=API_KEY,
        openai_api_base='https://qianfan.baidubce.com/v2/chat/completions',
        max_completion_tokens=8192,
        batch_size=2,
        run_cfg=dict(num_gpus=0),
        pred_postprocessor={
            'BYKJ-FEval-主观题*': dict(type=general_cn_postprocess),
            # '*': dict(type=first_capital_postprocess)
            },
        # default_headers={"appid":"app-s8p6lY9N"}
    )
]
