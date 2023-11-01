# Results

This markdown file aims to store the best results we obtain with the models at the different phases of training. The first phase of training consists before the improvements with the added column of last month and the redesign of the preprocessing pipeline.

## First phase

### sgd_regressor et linear_regressor

With both models we obtain the following results:
- mse : 159.8
- rmse : 12.6
- r2 : 0.115

### decision_tree_reg

With parameters 7 for the maximal depth and 8 for the minimum samples per leafes, we obtain the following results:
- mse : 144.4
- rmse : 12.0
- r2 : 0.200

### random_forest

With parameters 100 for the number of estimators, 7 for the maximal depth and 8 for the minimum samples per leaf, we obtain the following results:
- mse : 144.0
- rmse : 12.0
- r2 : 0.202

## Second phase
