import numpy as np
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import os
import bottleneck as bn
from scipy import sparse
import torch
from sklearn.decomposition import TruncatedSVD

def whitening_torch(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings


data = "ml-20m"
#data="netflix-prize"
#data="msd"

#
white=True
SVD=False

dir = "/home/onishi/recommend/" + data + "/pro_sg/"
model_dir=data+"_doc2model"
model = Doc2Vec.load( "/home/onishi/recommend/"+model_dir + "/model_1500_3000_ep20")
df = pd.read_csv(dir + "train.csv")
size = len(df['sid'].unique())
size_train=len(df['uid'].unique())
df_test=pd.read_csv(dir+"validation_tr.csv")
size_test=len(df_test['uid'].unique())

#pred=np.zeros((size_test,size))
user_id=df_test['uid'].unique()
user_id.sort()

#Create item embeddings
X=np.zeros((model.wv[str(1)].shape[0],size))
for i in range(size):
    X[:,i]=model.wv[str(i)]

#Create user embeddings
vectors=[model.dv[doc_id] for doc_id in user_id]
arr=np.array(vectors)
Y=arr.T

pred = cosine_similarity(Y.T,X.T)

#A pattern considering the similarity between user embeddings and item embeddings as a feature
"""
if white:
    x = torch.from_numpy(X.astype(np.float32)).clone()
    Y=whitening_torch(x.T)
    Y=Y.T
    X = Y.to('cpu').detach().numpy().copy()

similarity_matrix = cosine_similarity(X.T)
np.fill_diagonal(similarity_matrix, 0)
df['sid']=df['sid'].astype(str)

class Test:
    def __init__(self):
        #LabelEncoder()は文字列を数値に変えてくれる
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'uid'])
        items = self.item_enc.fit_transform(df.loc[:, 'sid'])
        return users, items
    def fit(self, df, implicit=True):
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()
        X = csr_matrix((values, (users, items)))
        self.X = X
df_test=pd.read_csv(dir+"validation_tr.csv")
df=pd.read_csv(dir+"train.csv")
model=Test()
model.fit(df)
users = df_test.loc[:, 'uid']
items = df_test.loc[:, 'sid']
u_enc = LabelEncoder()
users_id = u_enc.fit_transform(users)
items_id = model.item_enc.transform(items)
values = np.ones(df_test.shape[0])
shape = (u_enc.classes_.size, model.item_enc.classes_.size)
X = csr_matrix((values, (users_id, items_id)), shape=shape)
pred = X.dot(similarity_matrix)
"""
def NDCG(x_pred, x_test, k=100):
    user_num = x_pred.shape[0]
    print(user_num)
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
    print(idx_topk_part.shape)
    #ユーザ数と同じ行数の列ベクトルの作成とid_topk_partより最初のk列のインデックスを渡している
    topk_part = x_pred[np.arange(user_num)[:, np.newaxis],
            idx_topk_part[:, :k]]
    print(topk_part.shape)
    #これにより各行を降順にソート,順位順にインデックスを格納
    idx_part = np.argsort(-topk_part, axis=1)
    print(idx_part.shape)
    #ここに各ユーザ上位k個のアイテムのインデックスを格納（元のユーザー×アイテム集合における）
    idx_topk = idx_topk_part[np.arange(user_num)[:, np.newaxis], idx_part]
    print(idx_topk.shape)
    # ランキングごとの分母部分
    tp = 1. / np.log2(np.arange(2, k + 2))
    #ユーザごとに上位k個のアイテムに対する評価値を格納
    #x_test[np.arange(user_num)[:, np.newaxis],idx_topk]にはx_testの中に含まれる上位のアイテムのインデックス(idx_topk)を示す
    DCG = (x_test[np.arange(user_num)[:, np.newaxis],
        idx_topk].toarray() * tp).sum(axis=1)
    #各ユーザに対してk個のアイテムを推奨した場合の理想的な最大のDCGスコアを表す
    IDCG = np.array([(tp[:min(n, k)]).sum()
        for n in x_test.getnnz(axis=1)])
    #getnnz(axis=1)は各行の非0要素の数を返す。
    return DCG / IDCG
#推薦したアイテムがどれだけx_testの中に入っているかで評価
def Recall(x_pred, x_test, k):
    users_num = x_pred.shape[0]
    idx = bn.argpartition(-x_pred, k, axis=1)
    #x_predと同じサイズの全False行列作成
    X_pred_binary = np.zeros_like(x_pred, dtype=bool)
    #ここで推薦されたアイテムがあるインデックスをTrueに置き換え
    X_pred_binary[np.arange(users_num)[:, np.newaxis], idx[:, :k]] = True
    #x_testの要素が0でない要素をTrueにした配列
    X_true_binary = (x_test > 0).toarray()
    #同じ位置にTrueがある場合個数を入れる
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
            np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall
def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)
    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']
    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
        (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
        (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

unique_sid = list()
with open(os.path.join(dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())
n_items = len(unique_sid)
validation_data_tr, validation_data_te = load_tr_te_data(
        os.path.join(dir, 'validation_tr.csv'),
        os.path.join(dir, 'validation_te.csv'))


batch_size_test=2000
N_test=validation_data_tr.shape[0]
idx_list_test=range(N_test) 
test=validation_data_te[idx_list_test]
n100_list,r20_list,r50_list=[],[],[]
for bnum,st_idx in enumerate(range(0,N_test,batch_size_test)):
    end_idx=min(st_idx+batch_size_test,N_test)
    X = validation_data_tr[idx_list_test[st_idx:end_idx]]
    if sparse.isspmatrix(X):
        X = X.toarray()
        X = X.astype('float32')
        pred_val=pred[idx_list_test[st_idx:end_idx]]
        #すでに推薦したものを入れないようにしている
        pred_val[X.nonzero()]=-np.inf
        n100_list.append(NDCG(pred_val, test[st_idx:end_idx], k=100))
        r20_list.append(Recall(pred_val, test[st_idx:end_idx], k=20))
        r50_list.append(Recall(pred_val, test[st_idx:end_idx], k=50))
n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)

print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
