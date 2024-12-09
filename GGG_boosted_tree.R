library(bonsai)
library(lightgbm)
library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)

train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

gbm_recipe <- recipe(type~., data=train_data) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn=factor) %>%  ## Turn color to factor then dummy encode color
  step_lencode_glm(color, outcome = vars(type)) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

# Set up the workflow
workflow_model <- workflow() %>%
  add_recipe(gbm_recipe) %>%
  add_model(boost_model)

# Set up cross-validation
set.seed(123)  # For reproducibility
cv_folds <- vfold_cv(train_data, v = 5)

# Create a grid for hyperparameter tuning
grid <- grid_random(
  tree_depth(c(3, 10)),
  trees(c(100, 1000)),
  learn_rate(range = c(0.01, 0.1)),
  size = 20
)

# Run the tuning
tuned_results <- tune_grid(
  workflow_model,
  resamples = cv_folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

# Get the best hyperparameters
best_params <- tuned_results %>% 
  select_best(metric = "accuracy")

# Finalize the workflow with the best parameters
final_workflow <- finalize_workflow(workflow_model, best_params)

# Fit the final model on the training data
final_fit <- fit(final_workflow, data = train_data)

# Make predictions on the test data
boosted_preds <- predict(final_fit, new_data = test_data)

# Print predictions
print(nn_preds)
nn_preds

kaggle_submission <- boosted_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class)
#rename pred to count (for submission to Kaggle)
  vroom_write(x=kaggle_submission, file="./boostedPreds.csv", delim=",")
## This takes a few min (10 on my laptop) so run it on becker if you want

