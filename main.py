import pandas as pd

from ml import BayesRandomForestClassifier
from utils import load_mex_data, gen_validation_data, save_submission, RMSE

print "Generating validation data"
(X_train, Y_train, X_test, Y_test, X_labels) = gen_validation_data(num_data=10000, train_percent=99.9, ycols=['y1'], discrete_x_labels=10)

use_test_data = False

#print "Loading data"
#(X_train, Y_train, X_test, Y_test, X_labels) = load_mex_data(use_test_data=use_test_data, train_percent=50, ycols=[3], discrete_x_labels=15)

# X_train.info()
# Y_train.info()
# X_test.info()
# Y_test.info()
# X_labels.info()

for i in range(10,11,10):
    print "Training phase started"
    clf = BayesRandomForestClassifier(num_estimators=100, iterations=10, use_previous_features=False)
    clf.fit(X_train, Y_train)

    print "Prediction phase started"
    # Y_pred = clf.predict(X_test)
    logp = clf.eval(X_test, Y_test, num_points=10)
    print logp

#    if not use_test_data:
#        rmse = RMSE(Y_test, Y_pred)
#        print "Estimators: " + str(i) + " error: " + str(rmse)

save_submission(Y_test, filename='test.csv')

print "Saving prediction results to submission file"
#save_submission(Y_pred)
