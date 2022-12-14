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



def prepare_PFA(df,min_interactions_per_user=10):
    remove_nan_skills = True
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    if remove_nan_skills: 
      df = df[~df["skill_id"].isnull()]
    else:
      df.ix[df["skill_id"].isnull(), "skill_id"] = -1
    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]
    df["timestamp"] = df["start_time"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    #df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds() / (3600*24))
    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(inplace=True, drop=True)
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    item_skill = np.array(df[["item_id", "skill_id"]])
    for i in range(len(item_skill)):
      Q_mat[item_skill[i,0],item_skill[i,1]] = 1
    df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
    df["inter_id"] = df.index
    df = df[['user_id', 'item_id', 'timestamp', 'correct', "inter_id"]]
    df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
    df['correct'] = df['correct'].astype(np.int32)
    dict_q_mat = {i:set() for i in range(Q_mat.shape[0])}
    for elt in np.argwhere(Q_mat == 1):
      dict_q_mat[elt[0]].add(elt[1])
    X={}
    X['df'] = np.empty((0,5)) # Keep track of the original dataset
    for stud_id in df["user_id"].unique():
      df_stud = df[df["user_id"]==stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]].copy()
      df_stud.sort_values(by="timestamp", inplace=True) # Sort values 
      df_stud = np.array(df_stud)
      X['df'] = np.vstack((X['df'], df_stud))
    onehot = OneHotEncoder()
    X['users'] = onehot.fit_transform(X["df"][:,0].reshape(-1,1))
    X['items'] = onehot.fit_transform(X["df"][:,1].reshape(-1,1))
    sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),sparse.hstack([X['users'], X['items']])]).tocsr()
    return X,sparse_df
def apprentissage_PFA(X):
    all_users = np.unique(X[:,0].toarray().flatten())
    y = X[:,3].toarray().flatten()
    kf = KFold(n_splits=5, shuffle=True)
    splits = kf.split(all_users)
    for run_id, (i_user_train, i_user_test) in enumerate(splits):
      users_train = all_users[i_user_train]
      users_test = all_users[i_user_test]
      X_train = X[np.where(np.isin(X[:,0].toarray().flatten(),users_train))]
      y_train = X_train[:,3].toarray().flatten()
      X_test = X[np.where(np.isin(X[:,0].toarray().flatten(),users_test))]
      y_test = X_test[:,3].toarray().flatten()
      model = LogisticRegression(solver="liblinear", max_iter=400)
      model.fit(X_train[:,5:], y_train) # the 5 first columns are the non-sparse dataset
      y_pred_test = model.predict_proba(X_test[:,5:])[:, 1]
      l1={}
      l2={}
      l3={}
      ACC = accuracy_score(y_test, np.round(y_pred_test))
      l1.append(ACC)
      AUC = roc_auc_score(y_test, y_pred_test)
      l2.append(AUC)
      NLL = log_loss(y_test, y_pred_test)
      l3.append(NLL)
    return l1,l2,l3


