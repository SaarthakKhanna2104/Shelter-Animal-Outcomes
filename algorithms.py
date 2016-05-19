from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss

from pybrain.structure import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.structure.modules import SoftmaxLayer


def run_xgb(X_train,y_train,X_val,y_val):
	pass


def run_nn(ds):
	hidden_units = [50,80,100,120,150,200]
	
	net = buildNetwork(2127,50,5,bias=True,
					hiddenclass=SigmoidLayer,
					outclass=SoftmaxLayer)
	trainer = BackpropTrainer(net,ds,verbose=True,momentum=0.1,weightdecay=0.01)
	train_error, cv_error = trainer.trainUntilConvergence(maxEpochs=7,verbose=True)

	return net

def run_svm(X_train,y_train,X_val,y_val):
	kernels = ['rbf','linear','poly']
	best_score = float('inf')
	best_clf = None

	for kernel in kernels:
		clf = SVC(kernel=kernel,degree=2,probability=True,verbose=True)
		clf.fit(X_train,y_train)
		print "Predicting for cross validation .... \n\n"
		y_pred = clf.predict_proba(X_val)
		score = log_loss(y_val,y_pred)
		print "SCORE: %.4f with kernel: %s"%(score,kernel)
		if score < best_score:
			best_score = score
			best_clf = clf

	return best_clf

		
def run_random_forest(X_train,y_train,X_val,y_val):
	# n_estimators = [10,50,100,150,200,250,500,1000]
	n_estimators = [1000]
	best_score = float('inf')
	best_clf = None
	for estimators in n_estimators:
		clf = RandomForestClassifier(n_estimators=estimators)
		clf.fit(X_train,y_train)
		y_pred = clf.predict_proba(X_val)
		score = log_loss(y_val,y_pred)
		print "SCORE: %.5f with n_estimators: %d"%(score,estimators)
		if score < best_score:
			best_score = score
			best_clf = clf	

	# y_pred_val = forest.predict(X_val)
	# print classification_report(y_val,y_pred_val)
	# print "Accuracy: %.3f"%(accuracy_score(y_val,y_pred_val))

	return best_clf



runtime_options_dict = {1:run_random_forest,2:run_svm}
