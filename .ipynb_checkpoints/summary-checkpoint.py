import numpy as np
import torch.nn as nn

from collections import OrderedDict


def count_macs(module, input, output):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # for each conv output we need # kernel ops + bias
        # divided by number of output channels
        output_size = output.nelement()
        kernel_ops = module.weight.nelement()
        if module.bias is not None:
            kernel_ops += module.bias.nelement()

        return output_size * kernel_ops // module.out_channels

    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        if isinstance(module.kernel_size, tuple):
            kernel_ops = np.prod(module.kernel_size)
        else:
            kernel_ops = module.kernel_size ** (len(input[0].shape) - 2)
        return kernel_ops * output.nelement() // output.shape[1]

    elif isinstance(module, nn.Linear):
        batch_size = input[0].size(0)
        total_ops = module.in_features * module.out_features
        if module.bias is not None:
            total_ops += module.bias.nelement()
        return total_ops * batch_size

    elif isinstance(module, nn.LSTMCell):
        hidden_size = module.hidden_size
        input_size = module.input_size
        batch_size = input[0].size(0)

        total_ops = 0
        # f_t = sigmoid( W_f z + b_f)
        total_ops += hidden_size * \
            (input_size + hidden_size + 1) * batch_size
        # i_t = sigmoid(W_i z + b_i)
        total_ops += hidden_size * \
            (input_size + hidden_size + 1) * batch_size
        # g_t = tanh(W_C z + b_C)
        total_ops += hidden_size * \
            (input_size + hidden_size + 1) * batch_size
        # o_t = sigmoid (W_o z + b_t)
        total_ops += hidden_size * \
            (input_size + hidden_size + 1) * batch_size
        # C_t = f_t * C_t-1 + i_t * g_t
        total_ops += 2 * hidden_size * batch_size
        # h_t = o_t * tanh(C_t)
        total_ops += hidden_size * 2 * batch_size
        return total_ops

    return 0


def get_shape(tensors):
    if isinstance(tensors, (list, tuple)):
        shapes = [get_shape(t) for t in tensors]
        if len(shapes) == 1:
            return shapes[0]
        return shapes
    elif isinstance(tensors, dict):
        shapes = [get_shape(t) for _, t in tensors.items()]
        if len(shapes) == 1:
            return shapes[0]
        return shapes
    elif tensors is None:
        return []
    else:
        return list(tensors.shape)


def get_elements(tensors):
    if isinstance(tensors, (list, tuple)):
        total = 0
        for t in tensors:
            total += get_elements(t)
        return total
    elif isinstance(tensors, dict):
        total = 0
        for _, t in tensors.items():
            total += get_elements(t)
        return total
    elif tensors is None:
        return 0
    else:
        return tensors.nelement()


def summary(model, inputs):
    """ summary
    prints a summary of the passed model
    :param model: nn.Module with the model to be summarized
    :param inputs: tensor or TensorDict with inputs
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            summary[m_key]["input_shape"] = get_shape(input)
            summary[m_key]["output_shape"] = get_shape(output)

            summary[m_key]["num_inputs"] = get_elements(input)
            summary[m_key]["num_outputs"] = get_elements(output)

            params = 0
            for attr in dir(module):
                if attr.startswith("weight"):
                    weight = getattr(module, attr)
                    if hasattr(weight, "shape"):
                        params += np.prod(list(getattr(module, attr).shape))
                        summary[m_key]["trainable"] = weight.requires_grad
                if attr.startswith("bias"):
                    bias = getattr(module, attr)
                    if hasattr(bias, "shape"):
                        params += np.prod(list(bias.shape))
            summary[m_key]["num_params"] = params
            summary[m_key]["num_macs"] = count_macs(module, input, output)

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()

    # add "Input" layer
    summary["Input"] = OrderedDict()
    summary["Input"]["output_shape"] = get_shape(inputs)
    summary["Input"]["num_outputs"] = get_elements(inputs)
    summary["Input"]["num_params"] = 0
    summary["Input"]["num_macs"] = 0

    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    # format column widths
    layer_length = 20
    shape_length = 20
    for layer in summary:
        ll = len(layer)
        if ll + 2 > layer_length:
            layer_length = ll + 2
        sl = len(str(summary[layer]["output_shape"]))
        if sl + 2 > shape_length:
            shape_length = sl + 2
    line_length = layer_length + shape_length + 28

    print("-" * line_length)
    line_new = ("{:>" + str(layer_length) + "}  "
                "{:>" + str(shape_length) + "} {:>12} {:>12}")\
        .format("Layer (type)", "Output Shape", "Param #", "MACs #")
    print(line_new)
    print("=" * line_length)
    total_macs = 0
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = ("{:>" + str(layer_length) + "} "
                    "{:>" + str(shape_length) + "} {:>12} {:>12}")\
            .format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["num_params"]),
            "{0:,}".format(summary[layer]["num_macs"]),
        )
        total_params += summary[layer]["num_params"]
        total_macs += summary[layer]["num_macs"]
        total_output += summary[layer]["num_outputs"]
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] is True:
                trainable_params += summary[layer]["num_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = get_elements(inputs) * 4.0 / (1024 ** 2.0)
    # x2 for gradients
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2.0))
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    non_trainable_params = total_params - trainable_params
    print("=" * line_length)
    print("Total MACs: {0:,}".format(total_macs))
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(non_trainable_params))
    print("-" * line_length)
    print("Input size (MiB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MiB): %0.2f" % total_output_size)
    print("Params size (MiB): %0.2f" % total_params_size)
    print("Estimated Total Size (MiB): %0.2f" % total_size)
    print("-" * line_length)
