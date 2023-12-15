import joblib
import pandas as pd
import lightgbm as lgb
import mylib_stock2


# 各種設定
MODEL = "ENSEMBLE_GBM"
ENSEMBLE_NUM = 3 # < 4

# モデルの読み込み
if MODEL == "ENSEMBLE_GBM":
    models = [joblib.load(f"etc/models/model{i}.job") for i in range(ENSEMBLE_NUM)]
elif MODEL == "ENSEMBLE_GBM_LR":
    models = [joblib.load(f"etc/models/model1.job"), joblib.load(f"etc/models/model2.job")]
else:
    models = [joblib.load(f"etc/models/model.job")]
    
# データの読み込み(最新30日分ほどしか用意しない)
