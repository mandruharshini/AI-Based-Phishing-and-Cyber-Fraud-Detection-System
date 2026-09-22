"""Generate synthetic training data and save both RandomForest models."""
import os, random
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_extraction import URL_FEATURE_NAMES, MESSAGE_FEATURE_NAMES

SEED=42; random.seed(SEED); np.random.seed(SEED)
MODELS_DIR=os.path.join(os.path.dirname(__file__),'models'); os.makedirs(MODELS_DIR,exist_ok=True)
LABELS=['Safe','Suspicious','Fraud']

def make_data(n, kind):
    X=[]; y=[]
    for _ in range(n):
        c=random.choices([0,1,2],weights=[.55,.25,.20])[0]
        if kind=='url':
            if c==0: f=[np.random.randint(15,45),np.random.randint(1,3),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),0,0,np.random.choice([0,1],p=[.9,.1]),np.random.choice([0,1],p=[.95,.05]),0,np.random.randint(0,1),0,0,np.random.randint(0,20),np.random.randint(0,2),0]
            elif c==1: f=[np.random.randint(35,90),np.random.randint(2,4),np.random.randint(1,3),np.random.randint(1,3),np.random.randint(1,5),np.random.randint(1,4),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,3),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(5,40),np.random.randint(0,4),np.random.randint(0,2)]
            else: f=[np.random.randint(55,160),np.random.randint(3,6),np.random.randint(2,5),np.random.randint(2,6),np.random.randint(3,10),np.random.randint(3,8),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(2,6),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(20,80),np.random.randint(2,8),np.random.randint(0,2)]
        else:
            if c==0: f=[np.random.randint(20,200),np.random.randint(0,2),0,0,0,np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2)]
            elif c==1: f=[np.random.randint(40,300),np.random.randint(0,3),np.random.randint(0,3),np.random.randint(0,3),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,4),np.random.randint(0,4),np.random.randint(0,2),np.random.randint(0,2)]
            else: f=[np.random.randint(30,400),np.random.randint(1,4),np.random.randint(2,6),np.random.randint(1,5),np.random.randint(2,5),np.random.randint(0,2),np.random.randint(2,8),np.random.randint(2,8),np.random.randint(0,2),np.random.randint(0,2)]
        if random.random()<.06: c=random.randrange(3)
        X.append(f); y.append(c)
    return np.array(X,float),np.array(y,int)

def train(kind,names,out):
    X,y=make_data(6000,kind); xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    clf=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=3,random_state=SEED,class_weight='balanced'); clf.fit(xt,yt)
    print(classification_report(yv,clf.predict(xv),target_names=LABELS,zero_division=0))
    joblib.dump({'model':clf,'feature_names':names,'labels':LABELS},os.path.join(MODELS_DIR,out))

if __name__=='__main__':
    train('url',URL_FEATURE_NAMES,'url_model.joblib'); train('message',MESSAGE_FEATURE_NAMES,'message_model.joblib'); print('Models created successfully.')
