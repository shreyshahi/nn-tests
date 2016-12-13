library("ggplot2")
library("dplyr")

training_data = read.csv("./results/training_data.csv")
test_data = read.csv("./results/test_data.csv")


# select best models
best_models = training_data %>%
  group_by(depth, lr) %>%
  slice(which.min(error))
print(best_models)
