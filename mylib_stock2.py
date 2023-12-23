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
NOW_WEEK = NOW.weekday()
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


def add_price(df:pd.DataFrame) -> pd.DataFrame:
    # 予測結果のファイルにその後のデータを書き込む
    
    # もう追加してあるファイルには書き込まない
    if df.shape[1] >= 6:
        return None
    
    # Dateの日から1営業日経っていないファイルには書き込まない
    recorded_day = datetime.datetime.strptime(df['Date'][0], "%Y-%m-%d")
    if not day_n_far_biz(recorded_day, TODAY_LAGGED_DT, 1):
        return None
    
    # Dateから1営業日後の日付
    after1bizday_Date = afterNbizday_date(recorded_day, 1)
    
    # 念の為に分析日の日付を追記
    df['after1bizday_Date'] = after1bizday_Date
    
    # Dateの日から25営業日以上経ったものはErrorを記録
    if day_n_far_biz(recorded_day, TODAY_LAGGED_DT, 25):
        df['after1bizday_Open'] = "Error"
        df['after1bizday_Close'] = "Error"
        return df
    
    # クローリング
    codes = [str(code)[:-1] for code in df['Code']]
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
            open, close = "No_trade", "No_trade"
        # 取引があれば読み込む
        else:
            open, close = after1bizday_record[1], after1bizday_record[4]
            
        opens.append(open)
        closes.append(close)
    
    # dfにOpenとCloseを追記
    df['after1bizday_Open'] = opens
    df['after1bizday_Close'] = closes
    
    return df
    
    
def validate(df:pd.DataFrame) -> list:
    # 予測結果を受け取り，評価する
    
    # 日付
    date = df['Date'][0]
    
    # 記録(予測)から分析まで30日以上経ったデータはエラー
    if df['after1bizday_Open'][0] == "Error":
        return [df['Date'][0], "Error"]
    
    # 評価するのはNo_tradeじゃなかった銘柄のみ，評価する銘柄の割合
    sum_verify = len(df.query('after1bizday_Open != "No_trade"'))
    trade_rate = sum_verify/df.shape[0]
    
    # 取引がなかった銘柄は除く
    df = df[df['after1bizday_Open'] != "No_trade"]
    
    # ret1とret2の実際の値を計算
    df['ret1_real'] = (df['after1bizday_Close']/df['after1bizday_Open']) - 1.0
    df['ret2_real'] = (df['after1bizday_Close']/df['Close']) - 1.0
    
    # 予想の方向のみ
    hit1_rate = len(df.query('ret1_forecast * ret1_real > 0'))/sum_verify
    hit2_rate = len(df.query('ret2_forecast * ret2_real > 0'))/sum_verify
    
    # 上がると予想していた割合とそのうち当たった割合
    up1_rate = len(df.query('ret1_forecast>0'))/sum_verify
    up2_rate = len(df.query('ret2_forecast>0'))/sum_verify
    uphit1_rate =  len(df.query('ret1_forecast>0 and ret1_forecast*ret1_real>0'))/(up1_rate*sum_verify)
    uphit2_rate =  len(df.query('ret2_forecast>0 and ret2_forecast*ret2_real>0'))/(up2_rate*sum_verify)
    
    return [date, trade_rate, hit1_rate, hit2_rate, up1_rate, uphit1_rate, up2_rate, uphit2_rate]