local nn = require "nn"
local tnt = require "torchnet"

network = require "get_network"
iterator = require "get_iterator"

get_end_epoch_hook = function(depth, nonlin, drop, lr, f)
    hook = function(state)
        train_error = state.criterion.output
        to_write = string.format(
            "%d, %d, %s, %f, %f, %f\n",
            state.epoch,
            depth,
            nonlin,
            drop,
            lr,
            train_error
        )
        f:write(to_write)
    end
    return hook
end

get_end_training_hook = function(depth, nonlin, drop, lr, f, data)
    hook = function(state)
        for sample in data() do
            out = state.network:forward(sample.input)
            to_write = string.format(
                "%d, %s, %f, %f, %f, %f, %f\n",
                depth, nonlin, drop, lr,
                sample.input[1], sample.target[1], out[1]
            )
            f:write(to_write)
        end
    end
    return hook
end


depths = torch.range(1, 10):int()
nonlinearities = {"Sigmoid", "ReLU"}
dropout_probs = torch.range(0, 0.8, 0.1)
learning_rates = {0.001, 0.005, 0.01, 0.05, 0.1, 0.2}

training_out = io.open("results/training_data.csv", "w")
test_out = io.open("results/test_data.csv", "w")


for i, depth in pairs(depths:totable()) do
    for j, nonlin in pairs(nonlinearities) do
        for k, dropout in pairs(dropout_probs:totable()) do
            for l, learning_rate in pairs(learning_rates) do
                print("staring", depth, nonlin, dropout, learning_rate)
                net = network.get_network(depth, nonlin, dropout)
                data = iterator.get_train_data()
                criterion = nn.MSECriterion()
                engine = tnt.SGDEngine()
                engine.hooks.onEndEpoch = get_end_epoch_hook(
                    depth, nonlin, dropout,
                    learning_rate, training_out
                )
                engine.hooks.onEnd = get_end_training_hook(
                    depth, nonlin, dropout,
                    learning_rate, test_out,
                    iterator.get_test_data()
                )
                engine:train{
                    network=net,
                    iterator=data,
                    criterion=criterion,
                    lr=learning_rate,
                    maxepoch=25
                }
            end
        end
    end
end
