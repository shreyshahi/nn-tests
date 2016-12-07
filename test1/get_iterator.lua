local tnt = require "torchnet"
local csv = require "csvigo"


local iterator = {}


local data = {
    training = csv.load("train.csv"),
    test =  csv.load("test.csv"),
}


local get_dataset = function(mode)
    local dataset = tnt.Dataset()
    dataset.size = function(self)
        return #data[mode].x
    end
    dataset.get = function(self, idx)
        return {
            input = torch.Tensor({data[mode].x[idx]}),
            target = torch.Tensor({data[mode].y[idx]})
        }
    end
    return dataset
end


iterator.get_train_data = function()
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = 50,
            dataset = tnt.ShuffleDataset{
                dataset = get_dataset("training")
            }
        }
    }
end


iterator.get_test_data = function()
    return tnt.DatasetIterator{
        dataset = get_dataset("test")
    }
end

return iterator
