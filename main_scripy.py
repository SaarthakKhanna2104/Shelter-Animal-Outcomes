##### No Nan's: Datetime, Breed, Color, Animal Type, 
##### Nan's: Sexuponoutcome, Ageuponoutcome

import sys
import pandas as pd
import numpy as np
import feature_engineering as fe
from sklearn.cross_validation import train_test_split
import algorithms as algo
from algorithms import runtime_options_dict

from pybrain.datasets import ClassificationDataSet

def get_data(file_name):
	df = pd.read_csv(file_name,sep=',')
	return df


def get_train_test_split(train,train_outcome):
	X_train,X_val,y_train,y_val = train_test_split(train,train_outcome,test_size=0.15)
	return X_train,X_val,y_train,y_val


def run_classifier(X_train,y_train,X_val,y_val,algorithm):
	algo_to_run = runtime_options_dict[algorithm]
	clf = algo_to_run(X_train,y_train,X_val,y_val)
	return clf

def get_output_file(clf,train,train_outcome,test):
	clf.fit(train,train_outcome)
	y_pred = clf.predict_proba(test)
	results = pd.read_csv("sample_submission.csv")
	results['Adoption'],results['Died'],results['Euthanasia'], \
	results['Return_to_owner'],results['Transfer'] = y_pred[:,0],y_pred[:,1],y_pred[:,2],y_pred[:,3],y_pred[:,4]
	results.to_csv("submission.csv",index=False)


def get_ds_for_pybrain(X,y):
	ds = ClassificationDataSet(2127,nb_classes=5)
	tuples_X = [tuple(map(float,tuple(x))) for x in X.values]
	tuples_y = [tuple(map(float,(y,))) for y in y.values]
	for X,y in zip(tuples_X,tuples_y):
		ds.addSample(X,y)
	ds._convertToOneOfMany()
	return ds



def get_test_data_features_nn(X):
	tuples_X = [tuple(map(float,tuple(x))) for x in X.values]
	return tuples_X


def predict_output_nn(net,X_test):
	predictions = list()
	for tup in X_test:
		predictions.append(net.activate(tup))
	preds = np.asarray(predictions)
	ids = np.arange(1,len(X_test)+1).reshape(len(X_test),-1)
	output = np.append(preds,ids,axis=1)
	output = np.roll(output,1,axis=1)
	np.savetxt("submission.csv",output,"%d,%.8f,%.8f,%.8f,%.8f,%.8f",\
		header='Id,Adoption,Died,Euthanasia,Return_to_owner,Transfer',delimiter=',',comments="")

	


if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	algorithm = int(sys.argv[3])

	train_data = get_data(train_file)
	test_data = get_data(test_file)

	########### Convert Datetime to year, month and week #########
	fe.convert_date(train_data)
	fe.convert_date(test_data)

	########### Convert AgeuponOutcome to months ###########
	fe.convert_age_to_months(train_data)
	fe.convert_age_to_months(test_data)

	########### Inplace replacement of columns in the df ########
	train_id,test_id,train_outcome = fe.drop_columns(train_data,test_data)
	train,test = fe.get_dummy_data(train_data,test_data)
	# train_outcome = fe.get_dummy_data_y(train_outcome)


	print "Done converting data ... "

	if algorithm != 3:
		X_train,X_val,y_train,y_val = get_train_test_split(train,train_outcome)
		clf = run_classifier(X_train,y_train,X_val,y_val,algorithm)
		get_output_file(clf,train,train_outcome,test)
	else:
		ds = get_ds_for_pybrain(train,train_outcome)
		X_test = get_test_data_features_nn(test)
		clf = algo.run_nn(ds)
		predict_output_nn(clf,X_test)


