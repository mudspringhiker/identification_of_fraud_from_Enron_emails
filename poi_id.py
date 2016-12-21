#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# The following are all the original features (except for email_address).
features_list = ['poi',
				 'salary',
 				 'to_messages',
       			 'deferral_payments',
 				 'total_payments',
 				 'exercised_stock_options',
 				 'bonus',
 				 'restricted_stock',
 				 'shared_receipt_with_poi',
 				 'restricted_stock_deferred',
 				 'total_stock_value',
 				 'expenses',
 				 'loan_advances',
 				 'from_messages',
 				 'other',
 				 'from_this_person_to_poi',
 				 'director_fees',
 				 'deferred_income',
 				 'long_term_incentive',
 		 		 'from_poi_to_this_person'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)
# This part was lifted from the lesson on new features. 

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

# After some exploratory data analysis (plotting histograms of all 
# available features), I decided to remove the following:
# - deferral_payments
# - director_fees
# 
# The following email features were removed because the fractions
# were created above:
# - to_messages
# - from_messages
# - from_poi_to_this_person
# - from_this_person_to_poi
# The final feature_list is now:

features_list = ['poi', 'expenses', 'deferred_income', 'long_term_incentive', 
				 'fraction_from_poi', 'restricted_stock_deferred', 
				 'shared_receipt_with_poi', 'loan_advances', 'other', 'bonus', 
				 'total_stock_value', 'restricted_stock', 'salary', 'total_payments', 
				 'fraction_to_poi', 'exercised_stock_options']

### Store to my_dataset for easy export below.
my_dataset = data_dict

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

from sklearn import model_selection
features_train, features_test, labels_train, labels_test = \
 		model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

features_train = np.array(features_train)
features_test = np.array(features_test)
labels_train = np.array(labels_train)
labest_test = np.array(labels_train)

# I decided to perform my classification using a pipeline with a gridsearchcv and the following steps:
# a. feature scaling using MinMaxScaler
# b. dimensionality reduction using PCA
# c. buidling the classifier using Decision Trees, which I found to have the 
#	 highest accuracy, precision and recall

# Making the pipeline:

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(MinMaxScaler(), PCA(n_components=2), DecisionTreeClassifier())

# parameter grid for the DecisionTreeClassifier:
param_grid = {'decisiontreeclassifier__min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}

# gridsearch and cross-validation:
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

# fitting:
grid.fit(features_train, labels_train)

# evaluation metrics:
print "Test score: {:.2f}".format(grid.score(features_test, labels_test))
print "Best cross-validation accuracy: {:.2f}".format(grid.best_score_)
print "Best parameters: {}".format(grid.best_params_)
print ""

from sklearn.metrics import confusion_matrix, recall_score, precision_score
pred = grid.predict(features_test)
print "Confusion matrix: \n", confusion_matrix(pred, labels_test) 
print "Recall score: {:.2f}".format(recall_score(pred, labels_test))
print "Precision score: {:.2f}".format(precision_score(pred, labels_test))
from sklearn.metrics import classification_report
print "Classification report: \n", classification_report(pred, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = grid

CLF_PICKLE_FILENAME = 'my_classifier.pkl'
DATASET_PICKLE_FILENAME = 'my_dataset.pkl'
FEATURE_LIST_FILENAME = 'my_feature_list.pkl'

def main():
	dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
	main()

