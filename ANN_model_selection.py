import time
import numpy as np
import pandas as pd

import keras.backend as K
from keras import optimizers
from keras import regularizers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, make_scorer
import matplotlib.pyplot as plt

data = pd.read_csv('Chicago River Continuous.5.csv', header=None)

data = data.values

# Identify Input and Output Variables
X = data[:,:10]
y = data[:,10]

# nb of input parameters
N = np.shape(X)[1]
# Randomize Input Data
random_seed = 789
np.random.seed(random_seed)

perm = np.random.permutation(len(X))

# Randomly Shuffle the Dataset
X = X[perm]
y = y[perm]
print np.shape(X)
test_size = 400

X_train = X[:np.shape(X)[0]-test_size, :]
X_test  = X[-test_size:, :]

y_train = y[:np.shape(X)[0]-test_size]
y_test  = y[-test_size:]


# Build a scaler to standardize input data along the axis=0 (column)
scaler = StandardScaler().fit(X_train)

# Standardize train data
X_train = scaler.transform(X_train)

# Standardize test data
X_test = scaler.transform(X_test)

def r2(y_true, y_pred):
	"""define r squared value as metrics for model training
	"""
	# calculate the mean of the observation
	y_mean_obs = K.mean(y_true)
	
	SS_tot = K.sum((y_true - y_mean_obs)**2)
	# SS_reg = K.sum((y_pred - y_mean_obs)**2)
	
	SS_res = K.sum((y_true - y_pred)**2)
	r2 = 1. - (SS_res/SS_tot)
	return r2

# assign a name for the h5 file that stores model weights
ANN_weights_path = 'ANN_weights.h5'

#Defining the Model Architecture and Parameters
def ANN_model(nb_Hidden_Neurons=10, optimizer='sgd', init='glorot_uniform', activation='relu'):
	
	model = Sequential()
	model.add(Dense(nb_Hidden_Neurons, input_dim=N, kernel_initializer = init,
					kernel_regularizer = regularizers.l2(0.01), 
					activation=activation))
	model.add(Dense(1, kernel_initializer = init))
	
	# Compile model
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = [r2])
	return model

# define a grid search function that take model object as input
def Grid_Search_Training(model):
	
	r2_scorer = make_scorer(r2_score)
	
	# parameters grid
	activations = ['relu']
	optimizers = ['sgd','rmsprop']
	epochs = [250]
	batches = [8, 16]
	nb_Hidden_Neurons = [10, 20, 30, 40, 60]
	param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches,\
	                    nb_Hidden_Neurons=nb_Hidden_Neurons, activation=activations)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=r2_scorer)
	
	return grid

def run(grid_search = True): 

	if grid_search:
 
		t0 = time.time()

		model = KerasRegressor(build_fn = ANN_model, verbose=1)
		grid = Grid_Search_Training(model)
		print 'Start Training the model......'

		grid_result = grid.fit(X_train, y_train)
		print("Best R2 Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		
		t1 = time.time()
		t = t1-t0

		print 'The GirdSearch on ANN took %.2f mins.' %(round(t/60., 2))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))

	
	else:
		# create a model
		model = ANN_model()
		
		# serialize model to JSON
		model_json = model.to_json()
		with open("ANN_model.json", "w") as json_file:
			json_file.write(model_json)
		print model.summary()
		
		checkpointer = ModelCheckpoint(ANN_weights_path, verbose=1, save_best_only=True)
		history = model.fit(X_train, y_train, validation_split=0.2, epochs=150, batch_size=10, callbacks=[checkpointer])
		#Summarize History for Loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
		
run(grid_search=True)


# load json and create model
json_file = open('ANN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load model_weights
loaded_model.load_weights(ANN_weights_path)
y_pred = loaded_model.predict(X_test)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "R_squared value is:", r2_score(y_test, y_pred)
#Summarize y_pred and y_test
plt.plot(y_test, y_pred, 'ko', alpha=0.2)
plt.plot([0,6], [0,6], 'k')
plt.xlim(0, 6.)
plt.ylim(0, 6.)
plt.title('Predicted vs True')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.axhline(y=2.3)
plt.axvline(x=2.3)
plt.show()
