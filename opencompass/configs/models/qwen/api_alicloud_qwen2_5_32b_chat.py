
from opencompass.models import OpenAI
from opencompass.utils.text_postprocessors import text_after_think_tag, last_option_postprocess_regex

models = [
    dict(
        type=OpenAI,
        abbr='ali-bailian-deepseek-r1',
        path='qwen2.5-14b-instruct',
        key='ENV_ALICLOUD_KEY',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        max_completion_tokens=8192,
        batch_size=1,
        run_cfg=dict(num_gpus=0),
        pred_postprocessor={'*': dict(type=last_option_postprocess_regex,)}
    )
]
