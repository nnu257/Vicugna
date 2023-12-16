import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import joblib

def score(df, target='ret1', pred='y_pred'):   
    # 学習・評価データそれぞれにおけるtargetとpredの相関係数を計測
    corrcoef = np.corrcoef(df[target].fillna(0),df[pred].rank(pct=True, method="first"))[0,1]
    return corrcoef

def run_analytics(scores, figname):
    # 各統計値を計算
    print(f"Mean Correlation: {scores.mean():.4f}")
    print(f"Median Correlation: {scores.median():.4f}")
    '''
    print(f"Standard Deviation: {scores.std():.4f}")
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