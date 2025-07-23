import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import mylib_stock2_copy
from mylib_stock2_copy import NOW


# 追記された株価を検証し，成績を評価するプログラム

# 各種設定
NOW = NOW.strftime("%Y-%m-%d_%h-%m-%s")
INPUT_PATH = "datas/output"
OUTPUT_PATH = f"datas/results/result.csv"
GRAPH_PATH = f"datas/results/graph.png"
RET_THERES_INCLUDE_MIN = 0.007


# 予測結果のファイル一覧を取得
# 分析はret1sortのファイルのみでよい
filenames = [filename for filename in os.listdir(INPUT_PATH) if "ret1sort" in filename]

# 株価追記済みのファイルからdfに読み込み
dfs = []
for filename in filenames:
    df = pd.read_csv(f"{INPUT_PATH}/{filename}")
    if df.shape[1] >= 6:
        dfs.append(df)
        
# ret2の予測値の絶対値がRET_THERES_INCLUDE_MIN以上のレコードのみ残す
# なお，レコードのindexが採番されないと，mylib_stock2のvalidateで不具合が起こるので，再度採番する
dfs = [df[df['ret2_forecast'].abs() >= RET_THERES_INCLUDE_MIN].reset_index(drop=True) for df in dfs if "ret2_forecast" in df.columns]

# 実行ごとに評価して結果をリストに入れる
header = [["date", "hit1", "hit2", "up1", "uphit1", "up2", "uphit2", "down1", "downhit1", "down2", "downhit2","trade", "flag"]]
results = [mylib_stock2_copy.validate(df) for df in dfs]
results.sort(key=lambda x:x [0])
# 予測(forward.py)から検証(verify.py)までの間が30日以上(正確には25日)あくとErrorになるので，それを含む日は検証しない．
results = [[record[0]]+[f"{ele:.4f}" for ele in record[1:]] for record in results if record[1] != "Error"]
table = header + results

# 土日祝に実行すると，直近の株価データは前営業日の株価となる．
# しかし，実行日時は前営業日とは異なるため乱数が変化する．
# よって，「前営業日」の日付は同じだが，銘柄は異なるデータが登録できる．

# 記録
table = [",".join(record)+"\n" for record in table]
open(OUTPUT_PATH, "w").writelines(table)

# 記録をdfで読み込み直して，グラフに出力
df = pd.read_csv(OUTPUT_PATH)

# 実行ごとのhit率
# RET_THERESが設定されていれば，ret2の絶対値がTHERES以上のレコードのみ評価．その場合はhit1は参考程度に．
# 横軸がdate, 縦軸は各種予想のhit率
# 横軸には，dfのflagを元に☓をつける
labels = df["date"].copy()
x = list(range(len(labels)))
for i, flag in enumerate(df["flag"]):
    if flag == 1:
        labels[i] += "upx"
    elif flag == 2:
        labels[i] += "dox"
    elif flag == 3:
        labels[i] += "updox"
plt.figure(figsize=(13, 9))
plt.plot(x,df["uphit2"],label="uphit2")
#plt.plot(x,df["downhit2"],label="downhit2")
#plt.plot(x,df["hit2"],label="hit2")
#plt.plot(x,df_mean["uphit1"],label="uphit1")
#plt.plot(x,df_mean["downhit1"],label="downhit1")
#plt.plot(x,df_mean["hit1"],label="hit1")
plt.hlines([0.5], x[0], x[-1], linestyles='dashed')
plt.title(f"date and hit% with thereshold = {RET_THERES_INCLUDE_MIN}")
plt.xticks(x, df["date"])
plt.xticks(rotation=90)
plt.legend()
plt.savefig(GRAPH_PATH)
plt.clf()