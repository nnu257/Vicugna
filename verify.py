import os
import pandas as pd
import datetime

import mylib_stock2
from mylib_stock2 import NOW


# 追記された株価を検証し，成績を評価するプログラム

# 各種設定
NOW = NOW.strftime("%Y-%m-%d_%h-%m-%s")
INPUT_PATH = "datas/output"
OUTPUT_PATH = f"datas/results/{NOW}"


# 予測結果のファイル一覧を取得
# 分析はret1sortのファイルのみでよい
filenames = [filename for filename in os.listdir(INPUT_PATH) if "ret1sort" in filename]

# 株価追記済みのファイル
df_files = []
for filename in filenames:
    df_file = pd.read_csv(f"{INPUT_PATH}/{filename}")
    if df_file.shape[1] >= 6:
        df_files.append(df_file)
        
# 日付ごとに評価して結果をリストに入れる
results = [mylib_stock2.validate(df_file) for df_file in df_files]
print(results)

# 記録
#open(OUTPUT_PATH, "w").write("test")