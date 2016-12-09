local nn = require "nn"
local tnt = require "torchnet"

network = require "get_network"
iterator = require "get_iterator"
our_hooks = require "get_hooks"


depths = torch.range(1, 8):int()
nonlinearities = {"Sigmoid", "ReLU"}
dropout_probs = {0, 0.1, 0.2}
learning_rates = {0.001, 0.01, 0.1}

training_out = io.open("results/training_data.csv", "w")
test_out = io.open("results/test_data.csv", "w")


for i, depth in pairs(depths:totable()) do
    for j, nonlin in pairs(nonlinearities) do
        for k, dropout in pairs(dropout_probs) do
            for l, learning_rate in pairs(learning_rates) do
                print("staring", depth, nonlin, dropout, learning_rate)
                net = network.get_network(depth, nonlin, dropout)
                data = iterator.get_train_data()
                criterion = nn.MSECriterion()
                engine = tnt.SGDEngine()
                engine.hooks.onEndEpoch = our_hooks.get_end_epoch_hook(
                    depth, nonlin, dropout,
                    learning_rate, training_out
                )
                engine.hooks.onEnd = our_hooks.get_end_training_hook(
                    depth, nonlin, dropout,
                    learning_rate, test_out,
                    iterator.get_test_data()
                )
                engine:train{
                    network=net,
                    iterator=data,
                    criterion=criterion,
                    lr=learning_rate,
                    maxepoch=100
                }
            end
        end
    end
end
