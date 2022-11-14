# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz, hstack, csr_matrix


def prepare_BKT(data,student,kc):
  data=data[data['Student']==student]
  data=data[data[kc]==1]
  data=data[['Student','Correct',kc]]
  return data
def apprentissage_BKT(data):
  p_0=0.6
  p_t=0.3
  p_g=0.7
  p_s=0.5
  apprenti=list()
  correctness=list()
  for i in range(data.shape[0]):
    if(data['Correct'].iloc[i]==1):
      if(i==0):
        c=(p_0*(1-p_s))/((p_0*(1-p_s))+((1-p_0)*(p_g)))
        d=(c + (1-c)*p_t)
        apprenti.append(d)       
      else:
        c=(apprenti[i-1]*(1-p_s))/((apprenti[i-1]*(1-p_s))+((1-apprenti[i-1])*(p_g)))
        d=(c + (1-c)*p_t)
        apprenti.append(d)
    else:
      if(i==0):
        c=(p_0*(p_s))/((p_0*(p_s))+((1-p_0)*(1-p_g)))
        d=(c + (1-c)*p_t)
        apprenti.append(d)
      else:
        c=(apprenti[i-1]*(p_s))/((apprenti[i-1]*(p_s))+((1-apprenti[i-1])*(1-p_g)))
        d=(c + (1-c)*p_t)
        apprenti.append(d)
    correctness.append((apprenti[i]*(1-p_s))+((1-apprenti[i])*p_g))
  return apprenti , correctness
