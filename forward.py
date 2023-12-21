import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import pandas as pd
import sys

import mylib_stock2


# 株価の予測結果を検証(フォワードテスト)するプログラム

# 各種設定
OUTPUT_PATH = "datas/output_dummy"

NOW = datetime.datetime.now()
NOW_TIME = NOW.time()
TODAY_LAGGED = (NOW - datetime.timedelta(hours=16.5)).strftime('%Y-%m-%d')


# 営業時間+-マージンの時間は実行できない
# スクレイピングせずに予測だけ可能とする方法もあるが，スクレイピングできたかわかるようにするため不可能とする
START = datetime.time(8,30,0)
END = datetime.time(16,30,0)
if (START < NOW_TIME) and (NOW_TIME < END):
    print("8:30~16:30は実行できません．")
    sys.exit()

# 予測結果のファイル一覧を取得
filenames = [filename for filename in os.listdir(OUTPUT_PATH) if "DS_store" not in filename]
print(filenames)

# 検証用に結果を保存しておく
df_files = []

# 結果を読み込み，その後の株価記録(1日分)を追記，検証
for filename in filenames:
    
    # 読み込み
    print(filename)
    file_path = f"{OUTPUT_PATH}/{filename}"
    df_file = pd.read_csv(file_path)
    
    # データの追加
    df_file = mylib_stock2.add_price(df_file)

    # 検証用リストへの追加とcsvでの上書き保存
    if df_file:
        df_files.append(df_file)
        df_file.to_csv(file_path, index=False)
    
# 検証
if df_files:
    mylib_stock2.validate(df_files)
else:
    print("検証できるデータがありません．")
    
    
    
    
    
