import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import joblib
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

from mylib_biz import day_n_far_biz, afterNbizday_date


NOW = datetime.datetime.now()
NOW_TIME = NOW.time()
TODAY_LAGGED = (NOW - datetime.timedelta(hours=16.5)).strftime('%Y-%m-%d')
TODAY_LAGGED_DT = datetime.datetime.strptime(TODAY_LAGGED, "%Y-%m-%d")
START = datetime.time(8,30,0)
END = datetime.time(16,30,0)

DELAY = 2
KABUTAN_URL = "https://kabutan.jp/stock/kabuka?code="


def score(df, target, pred):   
    # 学習・評価データそれぞれにおけるtargetとpredの相関係数を計測
    corrcoef = np.corrcoef(df[target].fillna(0),df[pred].rank(pct=True, method="first"))[0,1]
    return corrcoef

def run_analytics(scores, figname):
    # 各統計値を計算
    print(f"Mean Correlation: {scores.mean():.4f}")
    print(f"Median Correlation: {scores.median():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    '''
    print(f"Mean Pseudo-Sharpe: {scores.mean()/scores.std():.4f}")
    print(f"Median Pseudo-Sharpe: {scores.median()/scores.std():.4f}")
    '''
    print(f'Hit Rate (% positive eras): {scores.apply(lambda x: np.sign(x)).value_counts()[1]/len(scores):.2%}\n')

    # rooling/era 相関係数をグラフ化
    scores.rolling(10).mean().plot(kind='line', title='Rolling Per Era Correlation Mean', figsize=(15,4))
    plt.axhline(y=0.0, color="r", linestyle="--")
    plt.savefig(f'datas/graphs/{figname}')
    plt.clf()
    
    
def binning(X_train, X_test, n_bins=5, encode='ordinal', strategy='quantile'):
    # ビニングモデルの初期化とfit, transform
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    discretizer.fit(X_train)
    
    # 予測用に保存
    joblib.dump(discretizer, "etc/discretizer.job")
    
    # 学習・評価データの特徴量を変換する
    X_train_binned = discretizer.transform(X_train)
    X_test_binned = discretizer.transform(X_test)
    
    return X_train_binned, X_test_binned


def add_price(df_file:pd.DataFrame) -> pd.DataFrame:
    # 予測結果のファイルにその後のデータを書き込む
    
    # もう追加してあるファイルには書き込まない
    if df_file.shape[1] >= 6:
        return None
    
    # Dateの日から1営業日経っていないファイルには書き込まない
    recorded_day = datetime.datetime.strptime(df_file['Date'][0], "%Y-%m-%d")
    if not day_n_far_biz(recorded_day, TODAY_LAGGED_DT, 1):
        return None
    
    # Dateから1営業日後の日付
    after1bizday_Date = afterNbizday_date(recorded_day, 1)
    
    # 念の為に分析日の日付を追記
    df_file['after1bizday_Date'] = after1bizday_Date
    
    # Dateの日から25営業日以上経ったものはErrorを記録
    if day_n_far_biz(recorded_day, TODAY_LAGGED_DT, 25):
        df_file['after1bizday_Open'] = "Error"
        df_file['after1bizday_Close'] = "Error"
        return df_file
    
    # クローリング
    codes = [str(code)[:-1] for code in df_file['Code']]
    responses = []
    for code in tqdm(codes, desc="Fetching responses..."):
        responses.append(requests.get(KABUTAN_URL + code))
        time.sleep(DELAY)
    
    # スクレイピング
    # codeのafter1bizday_Dateのopenとcloseを抽出
    opens, closes = [], []
    for code, response in zip(codes, responses):
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # 当日情報を取り出す
        today_table_body = soup.select(
            "#stock_kabuka_table > table.stock_kabuka0 > tbody")[0]
        date = ["20"+today_table_body.find("time").string.replace("/", "-")]
        values = [value.string.replace(",", "")
                    for value in today_table_body.find_all("td")[0:4]]
        volume = [today_table_body.find_all("td")[6].string.replace(",", "")]
        today_record = date + values + volume
        
        # メイン表の情報を取り出す
        main_table_body = soup.select(
            "#stock_kabuka_table > table.stock_kabuka_dwm > tbody")[0]
        main_table_records = main_table_body.find_all("tr")
        main_table_recors_value = []
        for record in main_table_records:
            date = ["20"+record.select("th > time")[0].string.replace(",", "").replace("/", "-")]
            values = [value.string.replace(",", "")
                        for value in record.find_all("td")[0:4]]
            volume = [record.find_all("td")[6].string.replace(",", "")]
            main_table_recors_value.append(date + values + volume)

        # 表すべて
        table = [today_record] + main_table_recors_value
        
        # after1bizday_Dateのレコードを検索
        after1bizday_record = [record for record in table if record[0] == after1bizday_Date.strftime("%Y-%m-%d")][0]
        
        # 対象の日のVolumeが0ならエラー
        if after1bizday_record[5] == "0":
            open, close = "Error", "Error"   
        # 取引があれば読み込む
        else:
            open, close = after1bizday_record[1], after1bizday_record[4]
            
        opens.append(open)
        closes.append(close)
    
    # df_fileにOpenとCloseを追記
    df_file['after1bizday_Open'] = opens
    df_file['after1bizday_Close'] = closes
    
    return df_file
    
    
def validate(df_files:list) -> None:
    # 予測結果のリストを受け取り，検証する
    # 記録(予測)から分析まで30日以上経ったデータは最後の列にErrorが入っている．
    # 取引なしの時もErrorとする．
    # 前者はエラー出力，後者はハンドリングできるよう実装すること．
    
    for df in df_files:
        pass