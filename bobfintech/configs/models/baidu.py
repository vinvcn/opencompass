
from opencompass.models import OpenAI
from opencompass.utils.text_postprocessors import general_cn_postprocess, first_capital_postprocess
API_KEY = 'bce-v3/ALTAK-peGyWiEbMkwhsxujzOPBL/38fee9479fe67b098bc54ab3cff2c7f9adcfef96'


from mmengine import Config, DictAction

Config.fromfile('configs/models/baidu.py')


models = [
    dict(
        type=OpenAI,
        abbr='ERNIE-X1-32K',
        path='ernie-x1-32k-preview',
        key=API_KEY,
        openai_api_base='https://qianfan.baidubce.com/v2/chat/completions',
        max_completion_tokens=8192,
        max_seq_len=8192,
        extra_body=dict(
            max_tokens=256
        ),
        batch_size=1,
        run_cfg=dict(num_gpus=0),
        pred_postprocessor={
            'BYKJ-FEval-主观题*': dict(type=general_cn_postprocess),
            # '*': dict(type=first_capital_postprocess)
            },
        # default_headers={"appid":"app-s8p6lY9N"}
    )
    ,
    dict(
        type=OpenAI,
        abbr='ERNIE-3.5',
        path='ernie-3.5-8k',
        key=API_KEY,
        openai_api_base='https://qianfan.baidubce.com/v2/chat/completions',
        max_completion_tokens=8192,
        batch_size=10,
        run_cfg=dict(num_gpus=0),
        pred_postprocessor={
            'BYKJ-FEval-主观题*': dict(type=general_cn_postprocess),
            # '*': dict(type=first_capital_postprocess)
            },
        # default_headers={"appid":"app-s8p6lY9N"}
    )
    ,
    dict(
        type=OpenAI,
        abbr='ERNIE-Speed-Pro',
        path='ernie-speed-pro-128k',
        key=API_KEY,
        openai_api_base='https://qianfan.baidubce.com/v2/chat/completions',
        max_completion_tokens=8192,
        batch_size=10,
        run_cfg=dict(num_gpus=0),
        pred_postprocessor={
            'BYKJ-FEval-主观题*': dict(type=general_cn_postprocess),
            # '*': dict(type=first_capital_postprocess)
            },
        # default_headers={"appid":"app-s8p6lY9N"}
    ),
]
