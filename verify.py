import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import mylib_stock2
from mylib_stock2 import NOW


# 追記された株価を検証し，成績を評価するプログラム

# 各種設定
NOW = NOW.strftime("%Y-%m-%d_%h-%m-%s")
INPUT_PATH = "datas/output"
OUTPUT_PATH = f"datas/results/result.csv"
GRAPH_PATH = f"datas/results/graph.png"
GRAPH_COMP_PATH = f"datas/results/graph_comp.png"


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
header = [["date", "hit1", "hit2", "up1", "uphit1", "up2", "uphit2", "down1", "downhit1", "down2", "downhit2","trade"]]
results = [mylib_stock2.validate(df_file) for df_file in df_files]
results.sort(key=lambda x:x [0])
results = [[record[0]]+[f"{ele:.4f}" for ele in record[1:]] for record in results]
table = header + results

# 土日祝に実行すると，金曜日や前日のデータ保存される．
# よって，日付は同じものの，乱数の違いにより銘柄が異なるデータが登録されている．
# つまり，同じ日付なのに銘柄が違うレコードがある．

# 記録
table = [",".join(record)+"\n" for record in table]
open(OUTPUT_PATH, "w").writelines(table)

# 記録をdfで読み込み直して，グラフに出力
df = pd.read_csv(OUTPUT_PATH)
df_mean = df.groupby("date", as_index=False).mean()
df_comp = df.query('date == "2024-01-12"')

# 一つ目のグラフは．横軸がdate, 縦軸はuphit1%, uphit2%, downhit1%, downhit2%
x = df["date"].unique()
y1 = df_mean["uphit1"]
y2 = df_mean["uphit2"]
y3 = df_mean["downhit1"]
y4 = df_mean["downhit2"]

plt.figure(figsize=(13, 9))
plt.plot(x,y1,label="uphit1")
plt.plot(x,y2,label="uphit2")
plt.plot(x,y3,label="downhit1")
plt.plot(x,y4,label="downhit2")
plt.title("date and %")
plt.xticks(rotation=90)
plt.legend()
plt.savefig(GRAPH_PATH)
plt.clf()

# 二つ目のグラフは，横軸が2024-01-12のn回目，縦軸が上記と同じ
y1 = df_comp["uphit1"]
y2 = df_comp["uphit2"]
y3 = df_comp["downhit1"]
y4 = df_comp["downhit2"]
x = np.arange(1,len(y1)+1,1)

plt.figure(figsize=(6, 4))
plt.plot(x,y1,label="uphit1")
plt.plot(x,y2,label="uphit2")
plt.plot(x,y3,label="downhit1")
plt.plot(x,y4,label="downhit2")
plt.title("2024-01-12")
plt.xticks(x)
plt.legend()
plt.savefig(GRAPH_COMP_PATH)
plt.clf()