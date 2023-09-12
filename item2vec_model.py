import numpy as np
import pandas as pd
from gensim.models import Word2Vec
data="ml-20m"
#data="netflix-prize"
#data="msd"
dir="/home/onishi/recommend/"+data+"/pro_sg/"

df=pd.read_csv(dir+"train.csv")
size=len(df['sid'].unique())
df['sid']=df['sid'].astype(str)

sentences=df.groupby('uid')['sid'].apply(list).tolist()
max_list = max(sentences, key=len)
max_size = len(max_list)
#validation_trの読み込み
df_test=pd.read_csv(dir+"validation_tr.csv")
#uidでソート
df_test.sort_values(by='uid')
df_test['sid']=df_test['sid'].astype(str)
test_list=df_test.groupby('uid')['sid'].apply(list).tolist()
print("test_list")
print(test_list[0])
user_id=df_test['uid'].unique()
user_id.sort()
test=[]
print(user_id[0])
i=0
print(len(test_list))
for list in test_list:
    list.append(str(user_id[i]))
    test.append(list)
    i+=1
print(test[0])
model_dir=data+"_model"
print(model_dir)
print(max_size)
sentences=sentences+test
print(len(sentences))


for  file_number in [1500,2000,3000]:                                          
    model=Word2Vec(sentences,sg=1,vector_size=file_number,window=3200,hs=0,negative=5,seed=0,min_count=1,sample=0,ns_exponent=-0.5,epochs=12)          
    model.save('/home/onishi/recommend/'+model_dir+'/id_model_{}_3200_best'.format(file_number))         
    print(file_number)
