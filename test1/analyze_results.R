library("ggplot2")
library("dplyr")

training_data = read.csv("./results/training_data.csv")
test_data = read.csv("./results/test_data.csv")

test_data$dropout = as.factor(test_data$dropout)

# select best models
best_models = training_data %>%
  group_by(depth) %>%
  slice(which.min(error))

best_sigmoids = training_data %>%
  filter(nonlin==" Sigmoid") %>%
  group_by(depth) %>%
  slice(which.min(error))

best_relus = training_data %>%
  filter(nonlin==" ReLU") %>%
group_by(depth) %>%
  slice(which.min(error))


for(dep in 1:8){
  model = best_models %>%
    filter(depth==dep)
  d = test_data %>%
    filter(depth == dep & lr==model$lr & nonlin==model$nonlin & dropout==model$dropout)
  p = ggplot(d)
  p = p + geom_line(aes(x=x, y=y), color="black", linetype="dashed")
  p = p + geom_line(aes(x=x, y=pred))
  fname = paste("./results/plots/best/",dep,".png", sep="")
  ggsave(plot=p, file=fname)
  p = ggplot(subset(d, abs(x) < 3))
  p = p + geom_line(aes(x=x, y=y), color="black", linetype="dashed")
  p = p + geom_line(aes(x=x, y=pred))
  fname = paste("./results/plots/best/",dep,"_zoomed.png", sep="")
  ggsave(plot=p, file=fname)
}

# sigmoids
for(dep in 1:8){
  model = best_sigmoids %>%
    filter(depth==dep)
  d = test_data %>%
    filter(depth == dep & lr==model$lr & nonlin==model$nonlin & dropout==model$dropout)
  p = ggplot(d)
  p = p + geom_line(aes(x=x, y=y), color="black", linetype="dashed")
  p = p + geom_line(aes(x=x, y=pred))
  fname = paste("./results/plots/sigmoids/",dep,".png", sep="")
  ggsave(plot=p, file=fname)
  p = ggplot(subset(d, abs(x) < 3))
  p = p + geom_line(aes(x=x, y=y), color="black", linetype="dashed")
  p = p + geom_line(aes(x=x, y=pred))
  fname = paste("./results/plots/sigmoids/",dep,"_zoomed.png", sep="")
  ggsave(plot=p, file=fname)
}


# ReLUs
for(dep in 1:8){
  model = best_relus %>%
    filter(depth==dep)
  d = test_data %>%
    filter(depth == dep & lr==model$lr & nonlin==model$nonlin & dropout==model$dropout)
  p = ggplot(d)
  p = p + geom_line(aes(x=x, y=y), color="black", linetype="dashed")
  p = p + geom_line(aes(x=x, y=pred))
  fname = paste("./results/plots/relus/",dep,".png", sep="")
  ggsave(plot=p, file=fname)
  p = ggplot(subset(d, abs(x) < 3))
  p = p + geom_line(aes(x=x, y=y), color="black", linetype="dashed")
  p = p + geom_line(aes(x=x, y=pred))
  fname = paste("./results/plots/relus/",dep,"_zoomed.png", sep="")
  ggsave(plot=p, file=fname)
}