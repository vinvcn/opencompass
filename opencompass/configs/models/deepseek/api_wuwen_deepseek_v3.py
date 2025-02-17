
from opencompass.models import OpenAI


models = [
    dict(
        type=OpenAI,
        abbr='api-wuwen-deepseek-v3',
        path='deepseek-v3',
        key='ENV_WUWEN_KEY',
        openai_api_base='https://cloud.infini-ai.com/maas/v1/chat/completions',
        max_completion_tokens=8192,
        batch_size=2,
        run_cfg=dict(num_gpus=0),
    )
]
