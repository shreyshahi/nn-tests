local hooks = {}

local get_model_name = function(depth, nonlin, drop, lr)
    return string.format(
        "results/models/%d_%s_%.2f_%.4f.torch", depth, nonlin, drop, lr)
end

hooks.get_end_epoch_hook = function(depth, nonlin, drop, lr, f)
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
        f:flush()
    end
    return hook
end

hooks.get_end_training_hook = function(depth, nonlin, drop, lr, f, data)
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
        torch.save(get_model_name(depth, nonlin, drop, lr), state.network)
    end
    return hook
end

return hooks