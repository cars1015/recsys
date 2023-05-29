import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data="ml-20m"
#data="netflix-prize"
#data="msd"
dir="/home/onishi/recommend/"+data+"/pro_sg/"

df=pd.read_csv(dir+"train.csv")
print(df.shape)
df_tr=pd.read_csv(dir+"validation_tr.csv")
size=len(df['uid'].unique())
size_tr=len(df_tr['uid'].unique())
#
df=pd.concat([df,df_tr])
df['sid']=df['sid'].astype(str)
#Consider users as sentence and items as words.
documents = []
sentence=df.groupby('uid')['sid'].apply(list).tolist()
for user_id, group in df.groupby('uid'):
    titles = group['sid'].tolist()
    document = TaggedDocument(words=titles, tags=[str(user_id)])
    documents.append(document)
for file_number in [1]:
    model = Doc2Vec(dm=1,vector_size=1200, min_count=1, epochs=20, window=3200,dm_concat=0,ns_exponent=0,sample=0.00001)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model_dir=data+"_doc2model"
    model.save('/home/onishi/recommend/'+model_dir+'/model_1200_3200_ep20_dm{}_con'.format(file_number))
    print(file_number)

