import joblib
import pandas as pd
import lightgbm as lgb
import requests
import time
from tqdm import tqdm
import sys
from bs4 import BeautifulSoup
import pprint


# 各種設定
MODEL = "ENSEMBLE_GBM"
ENSEMBLE_NUM = 3 # < 4
DO_SCRAPING = True
DELAY = 2
KABUTAN_URL = "https://kabutan.jp/stock/kabuka?code="


def fetch(url):
    # クロール関数
    return requests.get(url)


# モデルの読み込み
print("loading models...", end="", flush=True)
if MODEL == "ENSEMBLE_GBM":
    models = [joblib.load(f"etc/models/model{i+1}.job") for i in range(ENSEMBLE_NUM)]
elif MODEL == "ENSEMBLE_GBM_LR":
    models = [joblib.load(f"etc/models/model1.job"), joblib.load(f"etc/models/model2.job")]
else:
    models = [joblib.load(f"etc/models/model.job")]
print("finished!")


# 株価データのスクレイピング(最新30日分を用意)
codes_normal = joblib.load('etc/codes_normal.job')
codes_normal = codes_normal[0:2]

if DO_SCRAPING:
    # 株価データのリスト
    prices_normal_not_indices_30 = []
    
    # コードを結合したURLのレスポンスを取得
    print("fetching responses...")
    responses = []
    for code in tqdm(codes_normal):
        responses.append([fetch(KABUTAN_URL + str(code)[:-1]), code])
        time.sleep(DELAY)
    
    # 株価の上場廃止をチェック
    codes_anomaly = [response[1] for response in responses if "該当する銘柄は見つかりませんでした" in response[0].text]
    if codes_anomaly:
        print("以下のコードがスクレイピングされていません．上場廃止などを確認してください．")
        print(", ".join([str(code) for code in codes_anomaly]))
        
    # メイン表の日時を取り出すパターン
    pattern_mr = r">(.+?)</time>"
    
    # 株価などを取り出す
    for response, code in responses:
    
        soup = BeautifulSoup(response.text, "lxml")

        # 当日情報の表 本体
        today_table_body = soup.select(
            "#stock_kabuka_table > table.stock_kabuka0 > tbody")[0]

        # 当日情報を取り出す
        date = ["20"+today_table_body.find("time").string.replace("/", "-")]
        values = [value.string.replace(",", "")
                    for value in today_table_body.find_all("td")[0:4]]
        volume = [today_table_body.find_all("td")[6].string.replace(",", "")]
        today_record = date + values + volume

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

            value_record = date + values + volume
            main_table_recors_value.append(value_record)

        # メイン表すべて
        table = [today_record] + main_table_recors_value
        
        # 全体表に追加
        prices_normal_not_indices_30.append([table, code])
        

# modelのデータ形式に合わせてデータを整形
for i, code_prices in enumerate(prices_normal_not_indices_30):
    day_prices = code_prices[0]
    code = code_prices[1]
    
    for j, day_price in enumerate(day_prices):
        date, open, high, low, close, volume = day_price
        # レコードごとにデータを追加
        prices_normal_not_indices_30[i][0][j] = [0, date, code, open, high, low, close, 0, 0, volume, 0, open, high, low, close, volume]
    


pprint.pprint(prices_normal_not_indices_30)