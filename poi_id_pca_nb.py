#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop('TOTAL') # not a person
data_dict.pop('THE TRAVEL AGENCY IN THE PARK') # not a person
#data_dict.pop('BELFER ROBERT') # error in data
#data_dict.pop('BHATNAGAR SANJAY') # error in data
data_dict.pop('LOCKHART EUGENE E') # no data

### Fix data
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['total_stock_value'] = 0

data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290


### Create new feature(s)

def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """
    if poi_messages != 'NaN' or all_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)
    else:
        fraction = 0
    return fraction

for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)

    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    data_point["fraction_to_poi"] = fraction_to_poi

### Store dataset to my_dataset for easy export.
my_dataset = data_dict

### Put all features in a variable, all_features.

all_features = []
c = 0
for key in data_dict:
    if c < 1:
        for feature in data_dict[key]:
            all_features.append(feature)
        c += 1

### Make a list of all features to be removed.

features_remove = ["poi", "email_address", "from_poi_to_this_person", \
                   "from_this_person_to_poi", "from_messages", "to_messages"]

### Create "features_list", the features to be used to create the classifier.
### It must have "poi" as the first element.

features_list = ["poi"]
for feature in all_features:
    if feature not in features_remove:
        features_list.append(feature)

### Print the number of features to be used and what they are and the features not to be used.

print "features_list = {}".format(features_list)
print "Number of features: {}".format(len(features_list))
print "Features removed: {}".format(features_remove)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import StratifiedShuffleSplit
features = np.array(features)
labels = np.array(labels)
cv = StratifiedShuffleSplit(n_splits=1000, random_state=42)
for train_idx, test_idx in cv.split(features, labels):
    features_train, features_test = features[train_idx], features[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]

### Import modules

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

pipe = make_pipeline(MinMaxScaler(), PCA(random_state=42), GaussianNB())

print "Pipe steps: \n{}".format(pipe.steps)

# parameter grid for PCA and classifier:
param_grid = {'pca__n_components': range(2, 15)}

# gridsearch and cross-validation:
grid = GridSearchCV(pipe, param_grid=param_grid)

# fitting:
grid.fit(features_train, labels_train)

# evaluation metrics:
from sklearn.metrics import confusion_matrix, recall_score, precision_score, classification_report

print "Test score: {:.2f}".format(grid.score(features_test, labels_test))
#print "Best cross-validation accuracy: {:.2f}".format(grid.best_score_)
#print "Best parameters: {}".format(grid.best_params_)
pred = grid.predict(features_test)
print "Confusion matrix: \n{}".format(confusion_matrix(labels_test, pred))
print "Recall score: {:.2f}".format(recall_score(labels_test, pred))
print "Precision score: {:.2f}".format(precision_score(labels_test, pred))
print "Classification report: \n{}".format(classification_report(labels_test, pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = grid.best_estimator_
print clf

CLF_PICKLE_FILENAME = 'my_classifier.pkl'
DATASET_PICKLE_FILENAME = 'my_dataset.pkl'
FEATURE_LIST_FILENAME = 'my_feature_list.pkl'


def main():
	dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
	main()

