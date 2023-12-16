## 当プログラムの機能など
- このプログラムは，株価を予測するものです．
- 具体的には，n-30日~n-1日目までの株価を用いて，n日目の終値と始値の関係を予測します．
- 過去の株価データは，東京証券取引所のJ-Quantsを利用しました．
   - J-Quantsに無料会員登録すれば，2年分の株価データを利用できます．  
      (有料なら10年以上のデータを取得可)
   - なお，当プログラムは2年以上の期間でも対応できるようにコーディングしています．

## 売買対象
- 本プログラムの予測対象は日本株(REIT除く)です．

## 出力例
- 当プログラムの最終出力は，ユーザが指定した銘柄群の株価予測結果です．
- datas/result.txtです．  

## 特徴量
- 用いた特徴量は，以下の通りです．
   - 株価4本値
   - 出来高，売買代金
   - 移動平均
   - MACD, シグナル
   - RSI
   - サイコロジカル
   - 移動平均乖離率
   - ボリンジャーバンド
   - ストキャスティクス
   - スローストキャスティクス
   - モメンタム
   - 終値の階差数列
   - ボラティリティ

 
## 使い方
1. Pythonおよび必要ライブラリをダウンロード  
ライブラリは以下参考
```
!pip install matplotlib
!pip install japanize-matplotlib
!pip install pandas
!pip install tqdm
!pip install joblib
!pip install bs4
!pip install requests
!pip install lightgbm
!pip install matplotlib
!pip install sklearn
!pip isntall shap

```
2. 必要なディレクトリを作成
```
cd Vicugna
mkdir etc
cd etc
mkdir models
```
3. J-Quantsなどから株価データをダウンロードし，次節の形式に整形してetc/に置く  
*ダウンロードの方法については，私のリポジトリKagutuchiや，他の方のチュートリアルを参考に．  
Kagutuchiのpyファイルを実行すれば，自動で生成されますが，フォルダの依存関係はパスを修正してください．  
4. 株価予測
   1. learn_model.pyを実行
   2. forecast.pyを実行
```
python3 learn_model.py
python3 forecast.py
```


## 株価データの形式
- datas/price_expla.txtを参考に．
- 説明が大変なので，私のリポジトリKagutuchi内のファイルを実行して生成して欲しいです．


## 補足
- 当プログラムの使用は自己責任でお願いします．
- 用意するデータの形式がわからないなどあれば，Twitterまで連絡してもらえれば答えます．