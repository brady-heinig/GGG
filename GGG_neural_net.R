library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)
library(tensorflow)



# read in data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

nn_recipe <- recipe(type~., data=train_data) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn=factor) %>%  ## Turn color to factor then dummy encode color
  step_lencode_glm(color, outcome = vars(type)) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)),
                            levels=1)

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

folds <- vfold_cv(train_data, v = 5, repeats=1)

tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
bestTuneNN <- tuned_nn %>%
  select_best(metric = "accuracy")

# finalize workflow and make predictions
nn_model <- rand_forest(hidden_units = tuned_nn$hidden_units,
                        epochs = 50) %>%
  set_engine("keras") %>% 
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model) %>%
  fit(data=train_data)

nn_preds <- predict(nn_wf, new_data=test_data)

kaggle_submission <- nn_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred) %>% #Just keep datetime and prediction variables
  rename(type=.pred) %>% #rename pred to count (for submission to Kaggle)

vroom_write(x=kaggle_submission, file="./NNPreds.csv", delim=",")
## This takes a few min (10 on my laptop) so run it on becker if you want

