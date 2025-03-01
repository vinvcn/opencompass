from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-0.5b-instruct-hf',
        path='Qwen/Qwen2-0.5B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=0),
    )
]
