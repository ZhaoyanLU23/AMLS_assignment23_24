# Task A

## Objective

Binary classification task (using PneumoniaMNIST dataset). The objective
is to classify an image onto "Normal" (no pneumonia) or "Pneumonia"
(presence of pneumonia)

## TODOs:

* [x] Design model for task A
    * [x] Choose a model: [XGBoost](https://github.com/dmlc/xgboost)
    * [x] Understand [boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
    * [x] Try a demo of XGBoost
* [ ] Report training, validation, and testing errors / accuracies, along with describe any hyper-parameter tunice process.
    * [ ] [Cross Validation Reference](https://scikit-learn.org/dev/modules/cross_validation.html)
    * [x] Add tools to [estimate models](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
    * [x] [Get the best model and all metrics](https://xgboost.readthedocs.io/en/stable/python/examples/sklearn_examples.html#sphx-glr-python-examples-sklearn-examples-py) using [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
    * [ ] Try [early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping)
    * [x] Save and load [models](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)
    * [ ] Plot all figures needed

* The assessment will predominantly concentrate on how you articulate about the choice of models, how
you develop/train/validate these models, and how you report/discuss/analyse the
results.
