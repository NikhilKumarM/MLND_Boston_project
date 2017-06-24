# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer,fbeta_score
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X,y)
    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {
        'max_depth':[1,2,3,4,5,6,7,8,9,10]
    }
    y_predict =regressor.predict(X,y)
    print y_predict
    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric(y,y_predict))

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor,params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# when I call this function it gives error 
# reg = fit_model(X_train, y_train)

'''
TypeError                                 Traceback (most recent call last)
<ipython-input-35-192f7c286a58> in <module>()
      1 # Fit the training data to the model using grid search
----> 2 reg = fit_model(X_train, y_train)
      3 
      4 # Produce the value for 'max_depth'
      5 print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])

<ipython-input-34-633c9adeac13> in fit_model(X, y)
     29 
     30     # Fit the grid search object to the data to compute the optimal model
---> 31     grid.fit(X, y)
     32 
     33     # Return the optimal model after fitting the data

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/grid_search.pyc in fit(self, X, y)
    827 
    828         """
--> 829         return self._fit(X, y, ParameterGrid(self.param_grid))
    830 
    831 

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/grid_search.pyc in _fit(self, X, y, parameter_iterable)
    571                                     self.fit_params, return_parameters=True,
    572                                     error_score=self.error_score)
--> 573                 for parameters in parameter_iterable
    574                 for train, test in cv)
    575 

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __call__(self, iterable)
    756             # was dispatched. In particular this covers the edge
    757             # case of Parallel used with an exhausted iterator.
--> 758             while self.dispatch_one_batch(iterator):
    759                 self._iterating = True
    760             else:

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in dispatch_one_batch(self, iterator)
    606                 return False
    607             else:
--> 608                 self._dispatch(tasks)
    609                 return True
    610 

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in _dispatch(self, batch)
    569         dispatch_timestamp = time.time()
    570         cb = BatchCompletionCallBack(dispatch_timestamp, len(batch), self)
--> 571         job = self._backend.apply_async(batch, callback=cb)
    572         self._jobs.append(job)
    573 

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/externals/joblib/_parallel_backends.pyc in apply_async(self, func, callback)
    107     def apply_async(self, func, callback=None):
    108         """Schedule a func to be run"""
--> 109         result = ImmediateResult(func)
    110         if callback:
    111             callback(result)

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/externals/joblib/_parallel_backends.pyc in __init__(self, batch)
    324         # Don't delay the application, to avoid keeping the input
    325         # arguments in memory
--> 326         self.results = batch()
    327 
    328     def get(self):

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __call__(self)
    129 
    130     def __call__(self):
--> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
    132 
    133     def __len__(self):

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/cross_validation.pyc in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, error_score)
   1682 
   1683     else:
-> 1684         test_score = _score(estimator, X_test, y_test, scorer)
   1685         if return_train_score:
   1686             train_score = _score(estimator, X_train, y_train, scorer)

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/cross_validation.pyc in _score(estimator, X_test, y_test, scorer)
   1739         score = scorer(estimator, X_test)
   1740     else:
-> 1741         score = scorer(estimator, X_test, y_test)
   1742     if hasattr(score, 'item'):
   1743         try:

/home/hduser1/.local/lib/python2.7/site-packages/sklearn/metrics/scorer.pyc in __call__(self, estimator, X, y_true, sample_weight)
     96         else:
     97             return self._sign * self._score_func(y_true, y_pred,
---> 98                                                  **self._kwargs)
     99 
    100 

TypeError: 'numpy.float64' object is not callable


'''
