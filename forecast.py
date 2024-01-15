import joblib
import pandas as pd
import lightgbm as lgb
import requests
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import random
import sys
import os

import mylib_stock_copy
from mylib_stock2 import NOW_TIME, TODAY_LAGGED, START, END, DELAY, KABUTAN_URL


# 株価を予測するプログラム

# 各種設定
MODEL = "ENSEMBLE_GBM"
ENSEMBLE_NUM = 3 # < 4
SEED_RANDOMED = True
OUTPUT_PATH = "datas/output"
manual_codes = [30730,]

# 営業時間+-マージンの時間は実行できない
# スクレイピングせずに予測だけ可能とする方法もあるが，スクレイピングできたかわかるようにするため不可能とする
if (START < NOW_TIME) and (NOW_TIME < END):
    print("8:30~16:30は実行できません．")
    sys.exit()
    
# ランダムに抽出した銘柄について予測，記録
if SEED_RANDOMED:
    random.seed(NOW_TIME)
else:
    random.seed(TODAY_LAGGED)
    
# 銘柄リスト
print("loading codes...", end="", flush=True)
codes_normal = joblib.load('etc/codes_normal.job')
codes_normal = random.sample(codes_normal, 300) + manual_codes
print("finished!", flush=True)

# 株価データのスクレイピング(最新30日分を用意)        
# コードを結合したURLのレスポンスを取得
responses = []
for code in tqdm(codes_normal, desc="Fetching responses..."):
    responses.append([requests.get(KABUTAN_URL + str(code)[:-1]), code])
    time.sleep(DELAY)

# 銘柄コードの誤りと，ページが存在しない，かなり前に上場廃止した企業を除く
codes_error = [response[1] for response in responses if "該当する銘柄は見つかりませんでした" in response[0].text]

# codes_error内にある銘柄は削除, 出力は他のものとまとめて行う
if codes_error:
    codes_normal = [x for x in codes_normal if x not in codes_error]
    responses = [response for response in responses if response[1] not in codes_error]

# 株価などを取り出す
prices_normal_not_indices_30 = []
# codes_no_tradeは取引がない銘柄，codes_abolitionは上場廃止銘柄
codes_no_trade = []
codes_abolition = []
for response, code in responses:
    print(code, end=", ")
    
    # 30日のうち，1日でも取引高が0になっているものは除く
    no_trade = False

    soup = BeautifulSoup(response.text, "lxml")
    
    # 上場廃止を判定 (= 時価総額が"-"である)
    info_table_zika = soup.select(
        "#stockinfo_i3 > table > tbody > tr:nth-child(2) > td")[0]
    zika = info_table_zika.string
    
    if zika in ["－", "-"]:
        # codes_normalから該当銘柄は削除，以降の処理はスキップしてpricesには追加しない
        codes_abolition.append(code)
        del codes_normal[codes_normal.index(code)]
        continue

    # 当日情報の表 本体
    today_table_body = soup.select(
        "#stock_kabuka_table > table.stock_kabuka0 > tbody")[0]

    # 当日情報を取り出す
    date = ["20"+today_table_body.find("time").string.replace("/", "-")]
    values = [value.string.replace(",", "")
                for value in today_table_body.find_all("td")[0:4]]
    volume = [today_table_body.find_all("td")[6].string.replace(",", "")]
    today_record = date + values + volume
    
    if volume[0] == "0":
        no_trade = True

    # メイン表
    main_table_body = soup.select(
        "#stock_kabuka_table > table.stock_kabuka_dwm > tbody")[0]
    main_table_records = main_table_body.find_all("tr")

    # メイン表の各行から、要素を取り出す
    main_table_recors_value = []
    for record in main_table_records:

        date = ["20"+record.select("th > time")[0].string.replace(",", "").replace("/", "-")]
        values = [value.string.replace(",", "")
                    for value in record.find_all("td")[0:4]]
        volume = [record.find_all("td")[6].string.replace(",", "")]
        
        if volume[0] == "0":
            no_trade = True

        value_record = date + values + volume
        main_table_recors_value.append(value_record)

    # メイン表すべて
    table = [today_record] + main_table_recors_value
    
    # 全体表に追加
    prices_normal_not_indices_30.append([table, code])
    
    # 取引が一日でもなかったものは除く
    if no_trade:
        codes_no_trade.append(code)

print("") 

# codes_error内にある銘柄は削除してあるので，表示だけする
if codes_error:
        print("以下の銘柄は存在しないため，予測の対象外です．")
        print(", ".join([str(code) for code in codes_error])) 

# codes_no_trade内にある銘柄は削除
if codes_no_trade:
    print("以下の銘柄は取引高が0の日があるため，予測の対象外です．")
    print(", ".join([str(code) for code in codes_no_trade]))
    
    codes_normal = [x for x in codes_normal if x not in codes_no_trade]
    prices_normal_not_indices_30 = [x for x in prices_normal_not_indices_30 if x[1] not in codes_no_trade]

# codes_abolistion内にある銘柄は削除してあるので，表示だけする
if codes_abolition:
    print("以下の銘柄は上場廃止しているため，予測の対象外です．")
    print(", ".join([str(code) for code in codes_abolition]))

# modelのデータ形式に合わせてデータを整形
for i, code_prices in enumerate(prices_normal_not_indices_30):
    day_prices = code_prices[0]
    code = code_prices[1]
    
    for j, day_price in enumerate(day_prices):
        date, open, high, low, close, volume = day_price
        # レコードごとにデータを追加
        # カブタンから取得するため，予測に使用していないnum, limits, turnovervalue, adjfactorは0とし, 影響が小さいadjpricesはpricesとしている
        prices_normal_not_indices_30[i][0][j] = [0, date, code, open, high, low, close, 0, 0, volume, 0, 0, open, high, low, close, volume]
    
# price_normal_not_indices_30から余分な銘柄コードの情報を抜く
# [[[day_price, *m], code], *n] -> [[day_price, *m], *n]
for i in range(len(prices_normal_not_indices_30)):
    prices_normal_not_indices_30[i] = prices_normal_not_indices_30[i][0]

# データは日付の列以外，全てintかfloatにする
for i in tqdm(range(len(prices_normal_not_indices_30)), desc="Changing type..."):
    prices_normal_not_indices_30[i] = [[mylib_stock_copy.change_type(prices_normal_not_indices_30[i][j][k], k) for k in range(17)] for j in range(len(prices_normal_not_indices_30[i]))]

# カブタンは上から新しい順なので，並び替える
prices_normal_not_indices_30 = [list(reversed(x)) for x in prices_normal_not_indices_30]


# テクニカル指標の計算
# 本来はopenとhighの列だけでできるが，自作ライブラリを利用するため，形式を合わせて渡す
prices_normal_indices_30 = mylib_stock_copy.calculate_indices(codes_normal, prices_normal_not_indices_30)

# 各銘柄の最新日付のレコードのみ取得
for i in range(len(prices_normal_indices_30)):
    prices_normal_indices_30[i] = prices_normal_indices_30[i][-1]


# df化
columns=['Num', 'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'UpperLimit', 'LowerLimit', 'Volume', 'TurnoverValue', 'AdjustmentFactor', 'AdjustmentOpen', 'AdjustmentHigh', 'AdjustmentLow', 'AdjustmentClose', 'AdjustmentVolume', 'movingvolume_10', 'movingline_5', 'movingline_25', 'macd', 'signal', 'rsi_9', 'rsi_14', 'rsi_22', 'psycological', 'movingline_deviation_5', 'movingline_deviation_25', 'bollinger25_p1', 'bollinger25_p2', 'bollinger25_p3', 'bollinger25_m1', 'bollinger25_m2', 'bollinger25_m3', 'FastK', 'FastD', 'SlowK', 'SlowD', 'momentum_rate_10', 'momentum_rate_20', 'close_diff_rate1', 'close_diff_rate5', 'close_diff_rate25', 'volatility5', 'volatility25', 'volatility60', 'ret1_forecast', 'ret2_forecast', 'days_of_weeks', 'sector17', 'sector33', 'category']
df_real = pd.DataFrame(prices_normal_indices_30, columns=columns).fillna(0)

# 特徴量と目的パラメータの指定
features = ['Code',
            'Volume',
            'AdjustmentOpen',
            'AdjustmentHigh',
            'AdjustmentLow',
            'AdjustmentClose',
            'AdjustmentVolume',
            'movingvolume_10',
            'macd',
            'signal',
            'rsi_9',
            'rsi_14',
            'rsi_22',
            'psycological',
            'movingline_deviation_5',
            'movingline_deviation_25',
            'bollinger25_p1',
            'bollinger25_p2',
            'bollinger25_p3',
            'bollinger25_m1',
            'bollinger25_m2',
            'bollinger25_m3',
            'FastK',
            'FastD',
            'SlowK',
            'SlowD',
            'momentum_rate_10',
            'momentum_rate_20',
            'close_diff_rate1',
            'close_diff_rate5',
            'close_diff_rate25',
            'volatility5',
            'volatility25',
            'days_of_weeks',
            'sector17',
            'sector33']

# 予測データの選択
X_real = df_real[features]

# ビニング
discretizer = joblib.load("etc/discretizer.job")
X_real_binned = discretizer.transform(X_real)
X_real = pd.DataFrame(X_real_binned, index=X_real.index, columns=[f"{feat}_binned" for feat in features])


# モデルの読み込み
print("loading models...", end="", flush=True)
if MODEL == "ENSEMBLE_GBM":
    models1 = [joblib.load(f"etc/models/model_ret1_{i+1}.job") for i in range(ENSEMBLE_NUM)]
    models2 = [joblib.load(f"etc/models/model_ret2_{i+1}.job") for i in range(ENSEMBLE_NUM)]
elif MODEL == "ENSEMBLE_GBM_LR":
    models1 = [joblib.load(f"etc/models/model1_ret1.job"), joblib.load(f"etc/models/model2_ret1.job")]
    models2 = [joblib.load(f"etc/models/model1_ret2.job"), joblib.load(f"etc/models/model2_ret2.job")]
else:
    models1 = [joblib.load(f"etc/models/model_ret1.job")]
    models2 = [joblib.load(f"etc/models/model_ret2.job")]
print("finished!")

# モデルを用いた予測
if MODEL == "ENSEMBLE_GBM":
    tmp_real_pred1 = pd.Series([0]*len(X_real))
    tmp_real_pred2 = pd.Series([0]*len(X_real))
    
    # 複数のGBMのアンサンブル
    for i in range(ENSEMBLE_NUM):
        tmp_real_pred1 += models1[i].predict(X_real)
        tmp_real_pred2 += models2[i].predict(X_real)
        
    df_real[f'ret1_forecast'] = tmp_real_pred1/ENSEMBLE_NUM
    df_real[f'ret2_forecast'] = tmp_real_pred2/ENSEMBLE_NUM
    
# ファイル名の作成
filenames = [filename for filename in os.listdir(OUTPUT_PATH)]

for i in range(1,10000):
    filename1 = f"datas/output/{TODAY_LAGGED}_ret1sort{i}.csv"
    if filename1.split("/")[2] not in filenames:
        break

filename2 = filename1.replace("ret1", "ret2")
index = ["Date", "Code", "Close", "ret1_forecast", "ret2_forecast"]

# 予測結果の出力
df_real = df_real[index].sort_values(f"ret1_forecast", ascending=False)
df_real.to_csv(filename1, index=False)

df_real = df_real[index].sort_values(f"ret2_forecast", ascending=False)
df_real.to_csv(filename2, index=False)