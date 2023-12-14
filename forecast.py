import joblib

import pandas as pd

import time

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb
import shap

import mylib_stock2


# 各種設定
# DATA_USE_RATEは最大1で181万データ
SEED = 1234
MODEL = "GBM"
DATA_USE_RATE = 1
DO_SHAP = True

# 実行時間計測
start = time.time()

# 読み込み
print("now loading datas...", flush=True, end="")
prices_normal = joblib.load('etc/prices_normal.job')
print("finished!")

# データを分割して捨てる
prices_normal, not_use = train_test_split(prices_normal, train_size=DATA_USE_RATE-1e-5, random_state=SEED)

# 学習・評価データの分割(ランダムに銘柄で分割)
train, test = train_test_split(prices_normal, test_size=0.2, random_state=SEED)

# 銘柄でまとめてある配列を一次元化リスト化
flat_train = [x for row in train for x in row]
flat_test = [x for row in test for x in row]

# 各データセットをdf化
columns=['Num', 'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'UpperLimit', 'LowerLimit', 'Volume', 'TurnoverValue', 'AdjustmentFactor', 'AdjustmentOpen', 'AdjustmentHigh', 'AdjustmentLow', 'AdjustmentClose', 'AdjustmentVolume', 'movingvolume_10', 'movingline_5', 'movingline_25', 'macd', 'signal', 'rsi_9', 'rsi_14', 'rsi_22', 'psycological', 'movingline_deviation_5', 'movingline_deviation_25', 'bollinger25_p1', 'bollinger25_p2', 'bollinger25_p3', 'bollinger25_m1', 'bollinger25_m2', 'bollinger25_m3', 'FastK', 'FastD', 'SlowK', 'SlowD', 'momentum_rate_10', 'momentum_rate_20', 'close_diff_rate1', 'close_diff_rate5', 'close_diff_rate25', 'volatility5', 'volatility25', 'volatility60', 'ret1']
df_train = pd.DataFrame(flat_train, columns=columns).fillna(0)
df_test = pd.DataFrame(flat_test, columns=columns).fillna(0)


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
            #'psycological',
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
            'volatility60']

target = ['ret1']

# 学習・評価データの選択
X_train = df_train[features]
X_test = df_test[features]
y_train = df_train[target]
y_test = df_test[target]

# 金融時系列データは過分散のため，ビニングしてロバストにする
# KBinsDiscretizerを5分位等分布の設定で利用
X_train_binned, X_test_binned = mylib_stock2.binning(X_train, X_test)

# 新たにdfを上書き
X_train = pd.DataFrame(X_train_binned, index=X_train.index, columns=[f"{feat}_binned" for feat in features])
X_test = pd.DataFrame(X_test_binned, index=X_test.index, columns=[f"{feat}_binned" for feat in features])


# モデルの初期化と学習
if MODEL == "GBM":
    # LightGBM(決定木)
    model = lgb.LGBMRegressor(random_state=SEED, verbose=-1, max_depth=4)
    
    # 学習
    model.fit(X_train, y_train)
    
elif MODEL == "NEURAL":
    # パラメータ
    LAYERS = 1
    NODES = len(features)
    BATCH_SIZE = 64
    EPOCHS = 100
    
    # 深層ニューラルネットワーク
    model = keras.Sequential()
    for i in range(LAYERS):
        model.add(layers.Dense(NODES, activation='relu', input_shape=(len(features),)))
    model.add(layers.Dense(1))
    
    # モデルのコンパイル
    model.compile(optimizer='adam', loss='mean_squared_error')

    # モデルの学習
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)

# 訓練データでの予測値
df_train['y_pred'] = model.predict(X_train)
# テストデータでの予測値
df_test['y_pred'] = model.predict(X_test)


# 学習・評価データの分析結果を出力
print('--- Train Score ---')
train_scores = df_train.groupby('Date').apply(mylib_stock2.score)
mylib_stock2.run_analytics(train_scores, "train.png")

print('--- Test Score ---')
test_scores = df_test.groupby('Date').apply(mylib_stock2.score)
mylib_stock2.run_analytics(test_scores, "test.png")


# SHAPによる特徴量の重要度のグラフ化
if DO_SHAP:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("datas/graphs/features_shap.png")
    
# 実行時間
end = time.time()
print(end-start)