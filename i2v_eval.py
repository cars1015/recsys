from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
from scipy import sparse
import bottleneck as bn
import os
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
class EVAL:
    def NDCG(self, x_pred, x_test, k=100):
        user_num = x_pred.shape[0]
        idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
        topk_part = x_pred[np.arange(user_num)[:, np.newaxis],
                       idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(user_num)[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k + 2))
        DCG = (x_test[np.arange(user_num)[:, np.newaxis],
                        idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in x_test.getnnz(axis=1)])
        return DCG / IDCG
    def Recall(self, x_pred, x_test, k):
        users_num = x_pred.shape[0]
        idx = bn.argpartition(-x_pred, k, axis=1)
        X_pred_binary = np.zeros_like(x_pred, dtype=bool)
        X_pred_binary[np.arange(users_num)[:, np.newaxis], idx[:, :k]] = True
        X_true_binary = (x_test > 0).toarray()
        tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
        recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
        return recall
    def load_tr_te_data(self, csv_file_tr, csv_file_te):
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
class CORR:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'uid'])
        items = self.item_enc.fit_transform(df.loc[:, 'sid'])
        return users, items
    def whitening(self,embeddings):
        mu = np.mean(embeddings, axis=0, keepdims=True)
        cov = np.dot((embeddings - mu).T, embeddings - mu)
        u, s, vt = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1/np.sqrt(s)))
        embeddings = np.dot(embeddings - mu, W)
        return embeddings
    def fit_ease(self, df , X,lambda_: float = 0.5):
        users, items = self._get_users_and_items(df)

        G = X.dot(X.T)
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        print("inverse")
        P = np.linalg.inv(G)
        self.B = X.T.dot(P).dot(X)
        self.B[diagIndices] = 0
        # EASE
        #G[diagIndices] += lambda_
        #print("inverse")
        #P = np.linalg.inv(G)
        #B = P / (-np.diag(P))
        #B[diagIndices] = 0
    def fit_white(self, df , X, float = 0.5):
        users, items = self._get_users_and_items(df)
        X=self.whitening(X)
        B=cosine_similarity(X.T) 
        diagIndices = np.diag_indices(B.shape[0])
        B[diagIndices]=0
        self.B=B


data="ml-20m"
# data = "netflix-prize"
# data = "msd"
dir = "/home/onishi/recommend/" + data + "/pro_sg/"
df = pd.read_csv(dir + "train.csv")
model_dir=data+"_model"
model = Word2Vec.load( "/home/onishi/recommend/"+model_dir + "/model_600_3177_ens4")
size=len(df['sid'].unique())
X=np.zeros((model.wv[str(1)].shape[0],size))
for i in range(size):
    X[:,i]=model.wv[str(i)]


model = CORR()
#model.fit_ease(df,X,lambda_=14000.0)
model.fit_white(df,X)
df_test = pd.read_csv(dir + "test_tr.csv")
#
users = df_test.loc[:, 'uid']
items = df_test.loc[:, 'sid']
u_enc = LabelEncoder()
users_id = u_enc.fit_transform(users)
items_id = model.item_enc.transform(items)
# 
values = np.ones(df_test.shape[0])
shape = (u_enc.classes_.size, model.item_enc.classes_.size)
X = csr_matrix((values, (users_id, items_id)), shape=shape)
#
pred = X.dot(model.B)
eval = EVAL()
unique_sid = list()
with open(os.path.join(dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())
n_items = len(unique_sid)
test_data_tr, test_data_te = eval.load_tr_te_data(os.path.join(dir, 'test_tr.csv'), os.path.join(dir, 'test_te.csv'))
batch_size_test=2000
N_test=test_data_tr.shape[0]
idx_list_test=range(N_test)
test=test_data_te[idx_list_test]
n100_list,r20_list,r50_list=[],[],[]
for bnum,st_idx in enumerate(range(0,N_test,batch_size_test)):
    end_idx=min(st_idx+batch_size_test,N_test)
    X = test_data_tr[idx_list_test[st_idx:end_idx]]
    if sparse.isspmatrix(X):
        X = X.toarray()
        X = X.astype('float32')
    pred_val=pred[idx_list_test[st_idx:end_idx]]
    pred_val[X.nonzero()]=-np.inf
    n100_list.append(eval.NDCG(pred_val, test[st_idx:end_idx], k=100))
    r20_list.append(eval.Recall(pred_val, test[st_idx:end_idx], k=20))
    r50_list.append(eval.Recall(pred_val, test[st_idx:end_idx], k=50))
n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)
print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
