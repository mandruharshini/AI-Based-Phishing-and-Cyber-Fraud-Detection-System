"""
Loads trained models and provides prediction helpers.
"""
import os
import joblib
import numpy as np
from feature_extraction import extract_url_features, extract_message_features

MODELS_DIR=os.path.join(os.path.dirname(__file__),"models")
_url_bundle=None
_message_bundle=None

def _load_bundle(name):
    path=os.path.join(MODELS_DIR,name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{name}' not found. Run train_model.py first.")
    return joblib.load(path)

def _predict(bundle, vector, reasons):
    clf=bundle['model']; labels=bundle['labels']; proba=clf.predict_proba([vector])[0]
    idx=int(np.argmax(proba)); weights=np.array([0,50,100])[:len(proba)]
    return {'classification':labels[idx],'risk_score':round(float(np.clip(np.dot(proba,weights),0,100)),1),'confidence':round(float(proba[idx])*100,1),'class_probabilities':{l:round(float(p)*100,1) for l,p in zip(labels,proba)},'reasons':reasons}

def predict_url(url):
    global _url_bundle
    if not url or not url.strip(): raise ValueError('URL must not be empty')
    if _url_bundle is None: _url_bundle=_load_bundle('url_model.joblib')
    vector,reasons,raw=extract_url_features(url); result=_predict(_url_bundle,vector,reasons)
    result.update({'input_type':'url','input':url,'raw_features':raw}); return result

def predict_message(message):
    global _message_bundle
    if not message or not message.strip(): raise ValueError('Message must not be empty')
    if _message_bundle is None: _message_bundle=_load_bundle('message_model.joblib')
    vector,reasons,raw=extract_message_features(message); result=_predict(_message_bundle,vector,reasons)
    result.update({'input_type':'message','input':message,'raw_features':raw}); return result
