import pandas as pd
import numpy as np

path_to_data = '..\\mars-express-power-3years'


# Function to convert the utc timestamp to datetime
def convert_time(df):
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
    return df


# Function to resample the dataframe to hourly mean
def resample_1H(df):
    df = df.set_index('ut_ms')
    df = df.resample('1H', how='mean', closed=('right'))
    return df


# Function to read a csv file and resample to hourly consumption
def parse_ts(filename, dropna=True):
    df = pd.read_csv(path_to_data + '/' + filename)
    df = convert_time(df)
    df = resample_1H(df)
    if dropna:
        df = df.dropna()
    return df


# Function to read the ltdata files
def parse_ltdata(filename):
    df = pd.read_csv(path_to_data + '/' + filename)
    df = convert_time(df)
    df = df.set_index('ut_ms')
    return df


# Function to read the all data files and return training and test data
def load_mex_data(use_test_data = True, train_percent = 80.0,ycols = None, discrete_x_labels=0):
    ##Load the power files
    pow_train1 = parse_ts('/train_set/power--2008-08-22_2010-07-10.csv')
    pow_train2 = parse_ts('/train_set/power--2010-07-10_2012-05-27.csv')
    pow_train3 = parse_ts('/train_set/power--2012-05-27_2014-04-14.csv')
    # Load the test sample submission file as template for prediction
    if use_test_data:
        pow_test = parse_ts('power-prediction-sample-2014-04-14_2016-03-01.csv', False)
        # Concatenate the files
        power_all = pd.concat([pow_train1, pow_train2, pow_train3, pow_test])
    else:
        power_all = pd.concat([pow_train1, pow_train2, pow_train3])

    # Same for the saaf files
    saaf_train1 = parse_ts('/train_set/context--2008-08-22_2010-07-10--saaf.csv')
    saaf_train2 = parse_ts('/train_set/context--2010-07-10_2012-05-27--saaf.csv')
    saaf_train3 = parse_ts('/train_set/context--2012-05-27_2014-04-14--saaf.csv')
    if use_test_data:
        saaf_test = parse_ts('/test_set/context--2014-04-14_2016-03-01--saaf.csv')
        saaf_all = pd.concat([saaf_train1, saaf_train2, saaf_train3, saaf_test])
    else:
        saaf_all = pd.concat([saaf_train1, saaf_train2, saaf_train3])

    # Load the ltdata files
    ltdata_train1 = parse_ltdata('/train_set/context--2008-08-22_2010-07-10--ltdata.csv')
    ltdata_train2 = parse_ltdata('/train_set/context--2010-07-10_2012-05-27--ltdata.csv')
    ltdata_train3 = parse_ltdata('/train_set/context--2012-05-27_2014-04-14--ltdata.csv')
    if use_test_data:
        ltdata_test = parse_ltdata('/test_set/context--2014-04-14_2016-03-01--ltdata.csv')
        ltdata_all = pd.concat([ltdata_train1, ltdata_train2, ltdata_train3, ltdata_test])
    else:
        ltdata_all = pd.concat([ltdata_train1, ltdata_train2, ltdata_train3])

    # Extract the columns that need to be predicted
    power_cols = list(power_all.columns)

    # Now let's join everything together
    df = power_all

    # Make sure that saaf has the same sampling as the power, fill gaps with nearest value
    saaf_all = saaf_all.reindex(df.index, method='nearest')
    ltdata_all = ltdata_all.reindex(df.index, method='nearest')
    df = df.join(saaf_all)
    df = df.join(ltdata_all)

    # Now we formulate the prediction problem X -> Y
    # Y is the matrix that we want to predict
    # X is everything else
    if ycols == None:
        Y = df[power_cols]
    else:
        Y = df[[power_cols[i] for i in ycols]]
    X = df.drop(power_cols, axis=1)

    if discrete_x_labels > 0:
        (X, labels) = discretize(X, discrete_x_labels)

    # Splitting the dataset into train and test data
    if use_test_data:
        trainset = ~Y[power_cols[ycols[0]]].isnull()
    else:
        length = len(Y[power_cols[ycols[0]]])
        trainset = np.arange(length) < length * train_percent/100.0
    X_train, Y_train = X[trainset], Y[trainset]
    X_test, Y_test = X[~trainset], Y[~trainset]

    if discrete_x_labels > 0:
        return (X_train, Y_train, X_test, Y_test, labels)
    else:
        return (X_train, Y_train, X_test, Y_test)


# Function to save prediction to submission file
def save_submission(Y_test, filename='submission.csv'):
    # Converting the prediction matrix to a dataframe
    cols = Y_test.columns
    Y = pd.DataFrame(Y_test, index=Y_test.index, columns=cols)
    # We need to convert the parsed datetime back to utc timestamp
    Y['ut_ms'] = np.trunc(Y_test.index.astype(np.int64) * 1e-6)
    # Writing the submission file as csv
    cols = cols.insert(0, 'ut_ms')
    Y[cols].to_csv(filename, index=False)

# Defining the evaluation metric
def RMSE(val, pred):
    diff = (val - pred) ** 2
    rmse = np.mean(diff.values) ** 0.5
    return rmse

def binary_grid(n):
    if n == 1:
        return [[False], [True]]
    grid = binary_grid(n - 1)
    l = []
    for i in grid:
        l.append(i + [False])
        l.append(i + [True])
    return np.array(l)

def discretize(X, num_labels = 1):
    xmax = X.max()
    xmin = X.min()
    labels = pd.DataFrame(data=[xmin + i * (xmax - xmin)/float(num_labels) for i in range(num_labels)], columns=X.columns)
    Xd = np.trunc((X - xmin) / (xmax - xmin) * num_labels)
    Xd[Xd < 0] = 0
    Xd[Xd > num_labels-1] = num_labels-1
    return (Xd, labels)

def select(h,x,classes):
    l = []
    cl = list(classes)
    for i,c in enumerate(x.values):
        if c in cl:
            l.append(h[i,cl.index(c)])
        else:
            l.append(-np.inf)
    a = np.array(l)
    a[a < -10] = -10
    return a

def regularize(h):
    (r,c) = h.shape
    return 0.5*(h + np.ones((r,c))/float(c))

def gen_validation_data(num_data = 1000, num_columns = 10, train_percent = 80.0, ycols = None, discrete_x_labels = 0):

    trainset = np.arange(num_data) < num_data * train_percent/100.0
    data = np.array([np.random.random_sample(size=num_data) for i in range(num_columns)])
    X = pd.DataFrame(data=data.T, columns=['x' + str(i) for i in range(num_columns)])
    Y = X * (1.0 - X)
    Y['y1'] = Y[X.columns[:-1]].mean(axis=1)
    Y['y2'] = Y[X.columns[1:]].mean(axis=1)
    if ycols == None:
        Y = Y[['y1','y2']]
    else:
        Y = Y[ycols]

    if discrete_x_labels > 0:
        (X, labels) = discretize(X, discrete_x_labels)

    X_train, Y_train = X[trainset], Y[trainset]
    X_test, Y_test = X[~trainset], Y[~trainset]

    if discrete_x_labels > 0:
        return (X_train, Y_train, X_test, Y_test, labels)
    else:
        return (X_train, Y_train, X_test, Y_test)
