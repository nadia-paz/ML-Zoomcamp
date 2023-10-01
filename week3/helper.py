import pandas as pd 
import numpy as np 

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

drop_features = ['paymentmethod', 'multiplelines', 'gender', 'phoneservice', 'totalcharges']

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

def split_telco_data(df=get_telco_data(), seed=42, explore=True):
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    if explore:
        return df_train_full.reset_index(drop=True)
    else:
        df_train_full.drop(
            drop_features, 
            axis=1, 
            inplace=True)
        df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=seed)
        # get y arrays
        y_train = df_train.churn.values
        y_val = df_val.churn.values
        y_test = df_test.churn.values
        # delete target var from data sets
        del df_train['churn']
        del df_val['churn']
        del df_test['churn']

        return df_train.reset_index(drop=True), \
               df_val.reset_index(drop=True), \
               df_test.reset_index(drop=True), y_train, y_val, y_test

def get_numerical(explore=True):
    if explore:
        return ['tenure', 'monthlycharges', 'totalcharges']
    else:
        return ['tenure', 'monthlycharges']

def get_categorical(explore=True):
    cat = ['gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod'] 

    if explore:
        return cat
    else:
        return [c for c in cat if c not in drop_features]

def get_p_values(df: pd.DataFrame, cat_vars: list[str], alpha:float = 0.1):
    '''
    Performs Chi-squared test on categorical features of the dataframe.
    Parameters:
    df: data frame
    cat_vars: list of strings with names of categorical columns
    alpha: float, alpha value, default 0.1 for confidence interval 99%
    Returns a data frame with p_values of all categorical variables and their significance result
    '''

    #dictionary to hold names of the column and a p_value assotiated with it
    p_v = {}
    alpha = alpha
    #for every column in category variables run a chi2 test
    for col in cat_vars:
        #create a crosstable
        observed = pd.crosstab(df[col], df.churn)
        #run a chi squared test fot categorical data
        test = stats.chi2_contingency(observed)
        p_value = test[1]
        #add the result to the dictionary
        p_v[col] = p_value.round(3)
        
        #transform a dictionary to Series and then to Data Frame
        p_values = pd.Series(p_v).reset_index()
        p_values.rename(columns = {'index':'Feature', 0:'P_value'}, inplace = True)
        p_values = p_values.sort_values(by='P_value')

        #add the column that shows if the result is significant
        p_values['is_significant'] = p_values['P_value'] < alpha
    
    return p_values

def encode_dict_vect(train, val, test, get_dv = False):
    ''' 
    encodes train/val/test with DictVectorizer
    no first column drop
    '''
    # dataframes to lists of dictionaries
    train_dict = train.to_dict(orient='records')
    val_dict = val.to_dict(orient='records')
    test_dict = test.to_dict(orient='records')

    # create DictVectorizer
    dv = DictVectorizer(sparse=False)

    # fit on train, transform everything    
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    X_test = dv.transform(test_dict)

    if get_dv:
        return X_train, X_val, X_test, dv
    else:
        return X_train, X_val, X_test

def encode_one_hot(train, val, test):
    ''' 
    encodes train / val / test with OneHotEncoder
    drops the first categorical value
    '''
    # get categorical and numerical column names
    categorical = get_categorical(explore=False)
    numerical = get_numerical(explore=False)

    # create OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

    # fit on train tranform everything using categorical values only
    # concatenate with numerical columns
    X_train = np.concatenate(
    [
        ohe.fit_transform(train[categorical]),
        train[numerical]
    ], axis = 1)
    X_val = np.concatenate(
    [
        ohe.transform(val[categorical]),
        val[numerical]
    ], axis = 1)
    X_test = np.concatenate(
    [
        ohe.transform(test[categorical]),
        test[numerical]
    ], axis = 1)

    return X_train, X_val, X_test
    
