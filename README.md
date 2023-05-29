# recsys_MyReserch　　
推薦システムの研究で使う自分で作成、または引用、手直ししたコードを置く場所。　　
implicit feedback用のコード置き場
---
Preprocess.jpynb--ML-20M,Netflix-prize,MSDデータの前処理

EASE--線形回帰モデルEASEを用いて予測を作成
>関連論文:https://arxiv.org/pdf/1905.03375.pdf  
コード参考・引用元:https://github.com/Darel13712/ease_rec/blob/master/model.py 

Evaluation--予測結果をNDCG,Recallで評価　　
>関連論文:https://arxiv.org/pdf/1802.05814.pdf　　
コード参考・引用元；https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb　　

item2vec.jpyneb--word2vecを用いた推薦をimplicit feedbackに応用しモデルを作成　　
>関連論文:https://arxiv.org/ftp/arxiv/papers/1603/1603.04259.pdf  

i2v_evaluation.py--作成したモデルよりアイテムの埋め込みを取得しアイテム間類似度を用い推薦を行う。

