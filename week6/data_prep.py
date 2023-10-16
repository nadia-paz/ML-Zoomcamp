import numpy as np 
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

def credit_data():
    df = pd.read_csv('../data/credit.csv')
    df.columns = df.columns.str.lower()

    status_values = {
        1: 'ok',
        2: 'default',
        0: 'unk'
    }

    df.status = df.status.map(status_values)

    home_values = {
        1: 'rent',
        2: 'owner',
        3: 'private',
        4: 'ignore',
        5: 'parents',
        6: 'other',
        0: 'unk'
    }

    df.home = df.home.map(home_values)

    marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        0: 'unk'
    }

    df.marital = df.marital.map(marital_values)

    records_values = {
        1: 'no',
        2: 'yes',
        0: 'unk'
    }

    df.records = df.records.map(records_values)

    job_values = {
        1: 'fixed',
        2: 'partime',
        3: 'freelance',
        4: 'others',
        0: 'unk'
    }

    df.job = df.job.map(job_values)

    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace = 99999999, value = np.nan)

    df = df[df.status != 'unk'].reset_index(drop=True)

    return df

def split_data(df, full_train=False):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
    if full_train:
        return df_full_train
    else:
        df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

        df_full_train = df_full_train.reset_index(drop=True)
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        y_train = (df_train.status == 'default').astype('uint8').values
        y_val = (df_val.status == 'default').astype('uint8').values
        y_test = (df_test.status == 'default').astype('uint8').values

        del df_train['status']
        del df_val['status']
        del df_test['status']
        return df_train, df_val, df_test, y_train, y_val, y_test


#################################
df = credit_data()
df_full_train = split_data(df, full_train=True)
df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)

train_dicts = df_train.fillna(0).to_dict(orient='records')
val_dicts = df_val.fillna(0).to_dict(orient='records')
test_dicts = df_test.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)