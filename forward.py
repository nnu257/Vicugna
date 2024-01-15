import os
import pandas as pd
import sys

import mylib_stock2
from mylib_stock2 import NOW_WEEK, NOW_TIME, START, END


# 株価を予測した銘柄の1営業日後の株価をcsvに追記するプログラム

# 各種設定
OUTPUT_PATH = "datas/output"

# 平日は営業時間+-マージンの時間は実行できない
if (START <= NOW_TIME <= END) and (0 <= NOW_WEEK <= 4):
    print("平日の8:30~16:30は実行できません．")
    sys.exit()

# 予測結果のファイル一覧を取得
filenames = [filename for filename in os.listdir(OUTPUT_PATH) if "DS_store" not in filename]

# 結果を読み込み，その後の株価記録(1日分)を追記，検証
for i, filename in enumerate(filenames):
    
    # 読み込み
    print(f"{i+1}/{len(filenames)}　{filename}を分析中...")
    file_path = f"{OUTPUT_PATH}/{filename}"
    df_file = pd.read_csv(file_path)
    
    # データの追加
    df_file = mylib_stock2.add_price(df_file)

    # 上書き保存
    if not df_file is None:
        print("書き込み中です．終了しないでください...", flush=True, end="")
        df_file.to_csv(file_path, index=False)
        print("終了しました．", flush=True)

print("終了しました．")