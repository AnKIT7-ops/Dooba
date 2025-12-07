import numpy as np,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from collections import Counter
data={'Deadline':['Urgent','Urgent','Near','None','None','None','Near','Near','Near','Urgent'],
'Party':['Yes','No','Yes','Yes','No','Yes','No','No','Yes','No'],
'Lazy':['Yes','Yes','Yes','No','Yes','No','No','Yes','Yes','No'],
'Activity':['Party','Study','Party','Party','Pub','Party','Study','TV','Party','Study']}
df=pd.DataFrame(data)
le_dict={}
for col in df.columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    le_dict[col]=le
X=df[['Deadline','Party','Lazy']].values
y=df['Activity'].values
n=len(X);n_samples=5
classifiers=[]
for i in range(n_samples):
    idx=np.random.randint(0,n,n)
    t=DecisionTreeClassifier(max_depth=1)
    t.fit(X[idx],y[idx])
    classifiers.append(t)
def bagging_predict(X):
    preds=np.array([c.predict(X)for c in classifiers])
    final=[Counter(preds[:,i]).most_common(1)[0][0]for i in range(len(X))]
    return le_dict['Activity'].inverse_transform(final)
final_preds=bagging_predict(X)
y_enc=le_dict['Activity'].transform(final_preds)
acc=accuracy_score(y,y_enc)
prec=precision_score(y,y_enc,average='macro',zero_division=0)
rec=recall_score(y,y_enc,average='macro',zero_division=0)
f1=f1_score(y,y_enc,average='macro',zero_division=0)
print("Accuracy:",acc*100)
print("Precision:",prec)
print("Recall:",rec)
print("F1:",f1)
print("Correct:",le_dict['Activity'].inverse_transform(y))
print("Pred:",final_preds)
