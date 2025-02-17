import os, time
from huggingface_hub import snapshot_download

def download_llm(model_name, local_dir=None, **kwargs):
    snapshot_download(
        repo_id=model_name,
        max_workers=1,
        repo_type="model",
        local_dir="Qwen/Qwen2-0.5B-Instruct"
    )

if __name__ == "__main__":
    download_llm("Qwen/Qwen2-0.5B-Instruct")