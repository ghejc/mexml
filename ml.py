import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import binary_grid, select


class BayesRandomForestClassifier:
    def __init__(self, iterations=10, num_estimators=10, use_previous_features=False):
        self.classifiers = {}
        self.use_previous_features = use_previous_features
        self.iterations = iterations
        self.columns = []
        self.grid = []
        self.ymin = None
        self.ymax = None
        self.num_estimators = num_estimators

    def fit(self, x_train, y_train):
        self.ymin = y_train.min()
        self.ymax = y_train.max()
        self.columns = y_train.columns

        if self.use_previous_features:
            X = x_train[1:]
            X1 = x_train[:-1]
            Y = y_train[1:]
            X1.index = X.index
        else:
            X = x_train
            Y = y_train
        X = X.join(Y)
        for i in x_train.columns:
            clf = RandomForestClassifier(n_estimators=self.num_estimators, criterion="entropy")
            Y = X[i]
            X = X.drop([i], axis=1)
            if self.use_previous_features:
                clf.fit(X.join(X1[i]), Y)
            else:
                clf.fit(X, Y)
            self.classifiers[i] = clf

    def predict_logp(self, x_test, y):
        if self.use_previous_features:
            X = x_test[1:]
            X1 = x_test[:-1]
            Y = y[1:]
            X1.index = X.index
        else:
            X = x_test
            Y = y
        X = X.join(Y)
        logp = pd.Series(data = [0.0 for i in range(len(x_test))], name= 'logp', index=x_test.index)
        for i in x_test.columns:
            clf = self.classifiers[i]
            if clf == None:
                continue
            X = X.drop([i], axis=1)
            if self.use_previous_features:
                hist = clf.predict_log_proba(X.join(X1[i]))
                p = select(hist, x_test[i][1:], clf.classes_)
                logp = logp + pd.Series(data = p, name= 'logp', index=X.index)
            else:
                hist = clf.predict_log_proba(X)
                p = select(hist, x_test[i], clf.classes_)
                logp = logp + pd.Series(data= p, name = 'logp', index=x_test.index)
        return logp

    def predict(self, x_test):
        ymin = pd.DataFrame(data=[self.ymin for i in range(len(x_test))], index=x_test.index, columns=self.columns)
        ymax = pd.DataFrame(data=[self.ymax for i in range(len(x_test))], index=x_test.index, columns=self.columns)
        delta = 0.25 * (ymax - ymin)
        n = 0
        logpmax = pd.Series(data = [-np.inf for i in range(len(x_test))], name = 'logp',  index=x_test.index)
        ymean = 0.5 * (ymin + ymax)
        y = ymean.copy()
        yy = y.copy()
        grid = binary_grid(len(self.columns))
        while n < self.iterations:
            y1 = ymean - delta
            y2 = ymean + delta
            for a in grid:
                cols = self.columns[a]
                if len(cols) > 0:
                    y[cols] = y1[cols]
                cols = self.columns[np.invert(a)]
                if len(cols) > 0:
                    y[cols] = y2[cols]
                logp = self.predict_logp(x_test, y)
                rows = logp.values > logpmax.values
                if np.sum(rows) > 0:
                    logpmax[rows] = logp[rows]
                    yy[rows] = y[rows]
            ymean = yy
            delta = delta * 0.5
            n = n + 1
            # print "iter = " + str(n)
        return ymean

    def eval(self, x_test, y_test, num_points = 100):
        ymin = pd.DataFrame(data=[self.ymin for i in range(len(x_test))], index=x_test.index, columns=self.columns)
        ymax = pd.DataFrame(data=[self.ymax for i in range(len(x_test))], index=x_test.index, columns=self.columns)
        delta = (ymax - ymin) / float(num_points)
        n = 0
        y = ymin.copy()
        logp = pd.DataFrame(index=x_test.index)
        yy = pd.DataFrame(index=x_test.index)
        while n < num_points:
            logp['i' + str(n)] = self.predict_logp(x_test, y)
            yy['i' + str(n)] = np.abs(y - y_test)
            y = y + delta
            n = n + 1
            # print "iter = " + str(n)
        idx_pred = logp.idxmax(axis=1)
        idx_test = yy.idxmin(axis=1)
        result = pd.DataFrame(index=x_test.index)
        result["test"] = idx_test
        result["pred"] = idx_pred
        print result
        print 'predicting ' + str(np.sum(idx_test == idx_pred)) + ' of ' + str(len(idx_pred))
        return logp
