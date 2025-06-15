from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
# module for retriving datat 
from bayes_opt.util import load_logs

# import the function to be optimized
from classical_ml import *

# ------------------- random forest -------------------
# pbounds = {"batch_size" : (32, 128), "learning_rate" : (0.001, 0.009), "num_epochs": (1, 5)}
pbounds = {"n_estimators" : (10, 200), "max_depth" : (1, 20)}
def train_rf_wrapper(n_estimators, max_depth):
    n_estimators = int(round(n_estimators))
    max_depth = int(round(max_depth))
    accuracy = random_forest_classification(X_train, y_train, X_test, y_test, n_estimators, max_depth)
    return accuracy


# create instance of optimizer 
optimizer1 = BayesianOptimization(
    f = train_rf_wrapper,
    pbounds = pbounds,
    random_state = 1
)

# create UtilityFunction object for aqu. function
utility = UtilityFunction(kind = "ei", xi= 0.02)

# set gaussian process parameter
optimizer1.set_gp_params(alpha = 1e-6)

# create logger 
logger = JSONLogger(path = "./tunning_rf.log")
optimizer1.subscribe(Events.OPTIMIZATION_STEP, logger)

# initial search 
optimizer1.maximize(
    init_points = 5, # number of random explorations before bayes_opt
    n_iter = 15, # number of bayes_opt iterations
)

# print out the data from the initial run to check if bounds need update 
for i, param in enumerate(optimizer1.res):
    print(f"Iteration {i}: \n\t {param}")

# get best parameter
print("Best Parameters found: ")
print(optimizer1.max)






# -- ------------------- KNN -------------------
pbounds = {"n_neighbors" : (1, 20)}
def train_knn_wrapper(n_neighbors):
    n_neighbors = int(round(n_neighbors))
    accuracy = knn_classification(X_train, y_train, X_test, y_test, n_neighbors)
    return accuracy
# create instance of optimizer
optimizer2 = BayesianOptimization(
    f = train_knn_wrapper,
    pbounds = pbounds,
    random_state = 1
)

# create UtilityFunction object for aqu. function
utility = UtilityFunction(kind = "ei", xi= 0.02)

# set gaussian process parameter
optimizer2.set_gp_params(alpha = 1e-6)

# create logger 
logger = JSONLogger(path = "./tunning_knn.log")
optimizer2.subscribe(Events.OPTIMIZATION_STEP, logger)

# initial search 
optimizer2.maximize(
    init_points = 5, # number of random explorations before bayes_opt
    n_iter = 15, # number of bayes_opt iterations
)

# print out the data from the initial run to check if bounds need update 
for i, param in enumerate(optimizer2.res):
    print(f"Iteration {i}: \n\t {param}")

# get best parameter
print("Best Parameters found: ")
print(optimizer2.max)