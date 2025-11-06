import os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import login

# login(token="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")

api = HfApi()

from huggingface_hub import hf_hub_download, list_repo_files, HfApi
import os
import requests
import multiprocessing
from functools import partial
from tqdm import tqdm

REPO_ID = "compsciencelab/mdCATH"  # 这里改成你的仓库
LOCAL_DIR = '/inspire/ssd/project/sais-bio/public/tupeng/datasets/mdCATH'             # 保存路径
HF_MIRROR = "https://hf-mirror.com"     # 中国镜像（可选）
api = HfApi(endpoint=HF_MIRROR)

def download_data(f):
    filename = f.rfilename
    if os.path.exists(os.path.join(LOCAL_DIR, filename)):
        return None

    hf_hub_download(
            REPO_ID,
            filename,
            local_dir=LOCAL_DIR,
            repo_type="dataset",
            endpoint=HF_MIRROR,
            resume_download=True
        )
    
    return None

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    # 获取 repo 元信息（包含每个文件大小）
    info = api.repo_info(REPO_ID, repo_type="dataset")
    files = info.siblings

    with multiprocessing.Pool(processes = 16) as pool:
        partial_func = partial(download_data)
        results = list(tqdm(pool.imap(partial_func, files), total=len(files), desc="downloading mdCATCH...."))

    # for f in files:
    #     filename = f.rfilename
    #     if os.path.exists(os.path.join(LOCAL_DIR, filename)):
    #         continue

    #     size = getattr(f, "size", None)
    #     local_path = os.path.join(LOCAL_DIR, filename)
    #     os.makedirs(os.path.dirname(local_path), exist_ok=True)

    #     if size and file_complete(local_path, size):
    #         print(f"[SKIP] 已完成: {filename}")
    #         continue

    #     print(f"[DOWNLOAD] {filename}")
    #     hf_hub_download(
    #         REPO_ID,
    #         filename,
    #         local_dir=LOCAL_DIR,
    #         repo_type="dataset",
    #         endpoint=HF_MIRROR,
    #         resume_download=True
    #     )

if __name__ == "__main__":
    main()
