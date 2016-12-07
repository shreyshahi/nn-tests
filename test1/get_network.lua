local nn = require "nn"


local network = {}


local get_nonlinearity = function(nonlinearity)
    if nonlinearity == "Sigmoid" then
        return nn.Sigmoid()
    end
    return nn.ReLU()
end


network.get_network = function(num_hidden, nonlinearity, dropout_percent)
    dropout_percent = dropout_percent or 0.0
    net = nn.Sequential()
    net:add(nn.Linear(1, 5))
    net:add(get_nonlinearity(nonlinearity))
    net:add(nn.Dropout(dropout_percent))
    for i = 1,num_hidden do
        net:add(nn.Linear(5, 5))
        net:add(get_nonlinearity(nonlinearity))
        net:add(nn.Dropout(dropout_percent))
    end
    net:add(nn.Linear(5, 1))
    return net
end


return network
