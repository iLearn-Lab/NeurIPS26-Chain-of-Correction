from huggingface_hub import HfApi

api = HfApi()

print("开始分批上传海量文件...")
api.upload_large_folder(
    repo_id="yukiZhang0527/CoC-Artifacts",
    folder_path="/data/colosseum_dataset/failed_eval_demos",
    path_in_repo="failed_eval_demos",
    repo_type="dataset"
)
print("上传完成！")
