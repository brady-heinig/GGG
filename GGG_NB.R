library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)
library(discrim)
library(naivebayes)
library(themis)

# read in data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

# write recipe
my_recipe <- recipe(type ~ ., data=train_data) %>%
  step_mutate_at(color, fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%   # dummy variable encoding
  step_smote(all_outcomes(), neighbors=20)

## nb model3
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
# set up grid of tuning values
nb_tuning_params <- grid_regular(Laplace(),
                                 smoothness(),
                                 levels = 5)
# set up k-fold CV
folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_params,
            metrics=metric_set(roc_auc))

# find best tuning params
bestTuneNB <- CV_results %>%
  select_best(metric = "roc_auc")



# finalize workflow and make predictions
nb_model <- naive_Bayes(Laplace=bestTuneNB$Laplace, smoothness=bestTuneNB$smoothness) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) %>%
  fit(data=train_data)

nb_preds <- predict(nb_wf, new_data=test_data, type = "class")

kaggle_submission <- nb_preds %>%
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(type=.pred_class)  

vroom_write(x=kaggle_submission, file="./GGG_NBPreds.csv", delim=",")
