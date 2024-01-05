import os
import pandas as pd

import mylib_stock2
from mylib_stock2 import NOW


# 追記された株価を検証し，成績を評価するプログラム

# 各種設定
NOW = NOW.strftime("%Y-%m-%d_%h-%m-%s")
INPUT_PATH = "datas/output"
OUTPUT_PATH = f"datas/results/result.csv"


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
header = [["date", "hit1%", "hit2%", "up1%", "uphit1%", "up2%", "uphit2%", "down1%", "downhit1%", "down2%", "downhit2%","trade%"]]
results = [mylib_stock2.validate(df_file) for df_file in df_files]
results.sort(key=lambda x:x [0])
results = [[record[0]]+[f"{ele:.4f}" for ele in record[1:]] for record in results]
table = header + results

# 土日祝に実行すると，金曜日や前日のデータ保存される．
# よって，日付は同じものの，乱数の違いにより銘柄が異なるデータが登録されている．
# つまり，同じ日付なのに銘柄が違うレコードがある．

# 記録
table = [", ".join(record)+"\n" for record in table]
open(OUTPUT_PATH, "w").writelines(table)