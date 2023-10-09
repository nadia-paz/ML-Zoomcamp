import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

# parameters
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']
C = 1
n_splits = 5
model_file = f'model_C{C}.bin'

def get_telco_data():
    data = '../data/telco.csv'
    df = pd.read_csv(data)
    # 'coerce' fills errors with Null values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # fill nulls with zeros
    df.TotalCharges = df.TotalCharges.fillna(0)
    # rename columns
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # drop customer id
    # del df['customerid']
    # get names of string columns
    string_cols = df.dtypes[df.dtypes == 'object'].index
    # bring all values in string columns to lower case
    for col in string_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    # from yes/no to 1/0
    df.churn = (df.churn == 'yes').astype('uint8')

    return df

def get_train_test(df=get_telco_data(), seed=42):
    # drop_features = ['paymentmethod', 'multiplelines', 'gender', 'phoneservice', 'totalcharges']
    # df.drop(drop_features, axis=1, inplace=True)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    # get y arrays
    y_train = df_train.churn.values
    y_test = df_test.churn.values
    # delete target var from data sets
    del df_train['churn']
    del df_test['churn']

    return df_train.reset_index(drop=True),\
            df_test.reset_index(drop=True), y_train, y_test

def train_model(df, y, C=1.0):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

######### CREATE AND SAVE THE MODEL ###########
# copy from saving_model.ipynb

train, test, y_train, y_test = get_train_test()
print('Split data - done')
print()

dv, model = train_model(train, y_train, C=C)
print('Train model - done')
print()

print(f'Start model validation for C={C}')
scores = []
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for train_idx, val_idx in kfold.split(train):
    X_train = train.iloc[train_idx]
    X_val = train.iloc[val_idx]

    yt = y_train[train_idx]
    yv = y_train[val_idx]

    dv, model = train_model(X_train, yt)
    y_pred = predict(X_val, dv, model)
    
    auc = roc_auc_score(yv, y_pred)
    scores.append(auc)

print(f'Validation results')
print('%s %s %.3f +/- %.3f' % ('C=', C, np.mean(scores), np.std(scores)))
print()

y_hat = predict(test, dv, model)
print('Test prediction score: ', roc_auc_score(y_test, y_hat))
print()

with open(model_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print('Model saved to file ', model_file)