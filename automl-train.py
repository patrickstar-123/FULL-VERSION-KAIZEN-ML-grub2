import kagglehub
import pandas as pd
import os
import json
from ludwig.api import LudwigModel
from ludwig.automl import auto_train
from ray.tune import CheckpointConfig
import shutil
os.makedirs(r"D:\ray_tmp", exist_ok=True)
os.environ["RAY_TMPDIR"] = r"D:\ray_tmp"

# Ini untuk download dataset dari kaggle, kalau udah pernah download, dia bakal baca dari path.txt
# Membaca data langsung dari file lokal hasil unduhan manual
df = pd.read_csv('train.csv')

print("=== Pengecekan File Data ===")
print("Apakah data berhasil terbaca?", not df.empty)
print("Jumlah baris dan kolom data:", df.shape)
print("5 Baris data pertama:\n", df.head())
print("============================")


current_dir = os.getcwd()
output_directory = os.path.join(current_dir, "results")




current_dir = os.getcwd()
output_directory = os.path.join(current_dir, "results")

auto_train_results = auto_train(
    dataset=df,
    target="Survived",
    time_limit_s=7200,
    tune_for_memory=False,
    output_directory=output_directory,
    resume=False,
    skip_save_progress=True,
    user_config={
        "hyperopt": {
            "executor": {
                "num_samples": 2, 
                "max_concurrent_trials": 1, 
                "checkpoint_config": CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=5,
                ),
            }
        }
    },
)


print("AutoML training completed!")
# untuk ambil hasil terbaik dari auto_train, kita bisa ambil dari best_trial
analysis = auto_train_results.experiment_analysis
best_trial = analysis.best_trial
trial_result = best_trial.last_result
parameters = json.loads(trial_result["parameters"])
eval_stats = json.loads(trial_result["eval_stats"])
metrics = eval_stats["Survived"]

trial_model_directory = os.path.join(
    trial_result["trial_dir"], "hyperopt_ft_transformer", "model"
)
if not os.path.isdir(trial_model_directory):
    raise FileNotFoundError(
        f"Best trial model files were not found at {trial_model_directory}"
    )

best_model = LudwigModel.load(trial_model_directory)
model_directory = os.path.join(output_directory, "best_model")
best_model.save(model_directory)
print(f"Best model saved to: {model_directory}")

print(f"\nBest trial: {best_trial.trial_id}")
print(f"Validation ROC-AUC: {trial_result['metric_score']:.4f}")
print("Tuned parameters:")
for name, value in parameters.items():
    print(f"  {name}: {value}")

print("Validation metrics:")
for name in ("accuracy", "roc_auc", "precision", "recall"):
    print(f"  {name}: {metrics[name]:.4f} ({metrics[name]:.2%})")


# ray_temp = os.path.join(os.environ["LOCALAPPDATA"], "ray_tmp")
# os.makedirs(ray_temp, exist_ok=True)

# if not ray.is_initialized():
#     ray.init(_temp_dir=ray_temp, include_dashboard=False)