import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import pandas as pd
import sys

import mylib_stock2
from mylib_stock2 import NOW, NOW_TIME, TODAY_LAGGED, START, END


# 株価の予測結果を検証(フォワードテスト)するプログラム

# 各種設定
OUTPUT_PATH = "datas/output"


# 営業時間+-マージンの時間は実行できない
if (START < NOW_TIME) and (NOW_TIME < END):
    print("8:30~16:30は実行できません．")
    sys.exit()

# 予測結果のファイル一覧を取得
filenames = [filename for filename in os.listdir(OUTPUT_PATH) if "DS_store" not in filename]

# 結果を読み込み，その後の株価記録(1日分)を追記，検証
for filename in filenames:
    
    # 読み込み
    print(f"{filename}を分析中...")
    file_path = f"{OUTPUT_PATH}/{filename}"
    df_file = pd.read_csv(file_path)
    
    # データの追加
    df_file = mylib_stock2.add_price(df_file)

    # 検証用リストへの追加とcsvでの上書き保存
    if not df_file is None:
        df_file.to_csv(f"{OUTPUT_PATH}/{filename}", index=False)
        
print("終了しました．")