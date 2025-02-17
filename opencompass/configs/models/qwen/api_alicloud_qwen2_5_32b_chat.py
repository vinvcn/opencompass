
from opencompass.models import OpenAISDK


models = [
    dict(
        type=OpenAISDK,
        abbr='ali-bailian-deepseek-r1',
        path='qwen2.5-32b-instruct',
        key='ENV_ALICLOUD_KEY',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        max_completion_tokens=8192,
        batch_size=2,
        run_cfg=dict(num_gpus=0),
    )
]
