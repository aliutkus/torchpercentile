import torch


class Percentile(torch.autograd.Function):
    """
    torch Percentile autograd Functions subclassing torch.autograd.Function

    computes the percentiles of a tensor over the first axis
    """
    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    # def __init__(self, nb_percentiles):
    #     """ Inits a Percentile object with `nb_percentiles` regularly spaced
    #     between 0 and 100"""
    #     self.nb_percentiles = nb_percentiles
    #     self.percentiles = torch.linspace(0, 100, nb_percentiles)

    def forward(self, input, percentiles):
        """
        Find the percentiles of a tensor along the first dimension.
        """
        input_shape = input.shape
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        in_sorted = in_sorted.double()
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        return (d0+d1).view(-1, *input_shape[1:])

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input
