# recsys
推薦システムの研究で使う自分で作成、または引用、手直ししたコードを置く場所。
https://arxiv.org/abs/2308.13536
implicit feedback用のコード置き場
---
Preprocess.jpynb--ML-20M,Netflix-prize,MSDデータの前処理

EASE.py--線形回帰モデルEASEおよびAutoEncoder用いて予測を作成、評価
>関連論文:https://arxiv.org/pdf/1905.03375.pdf  
コード参考・引用元:https://github.com/Darel13712/ease_rec/blob/master/model.py、https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb 


item2vec_model.py--word2vecを用いた推薦（item2vec）をimplicit feedbackに応用しモデルを作成　　
>関連論文:https://arxiv.org/ftp/arxiv/papers/1603/1603.04259.pdf  

i2v_eval.py--作成したモデルよりアイテムの埋め込みを取得しアイテム間類似度を用い推薦を行う。　

Doc2_model.py--userを文、アイテムを単語とみなしDoc2Vecを用いた推薦モデルの作成    

Doc2_evaluation.py--Doc2vecを使用したモデルの評価

使用環境
---
jpynbファイルはGoogle colab    
pyファイルはlinux
Python 3.9.13
