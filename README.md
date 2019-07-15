# torchpercentile
## Percentile computation for pytorch

## Presentation

This repository implements a `Percentile` class that overrides torch.autograd.Function, enabling
percentile computation on the GPU for Pytorch.

```
class Percentile(torch.autograd.Function):
    def forward(ctx, input, percentiles)
```

This function returns the percentiles of the input, _computed along the first dimension_.

It works similarly to the numpy.percentile function, except that we don't accept a selection of axis, computations are done on the first axis only.
Note however that the shape of the tensor may be arbitrary.

### Parameters for `Percentile.forward`

* `input` : a Pytorch Tensor:
The data for which percentiles will be computed

* `percentiles` : a Pytorch Tensor, or some type that may be called as a parameter to `torch.tensor`
The different percentiles (between 0 and 100) to compute for the data

### Results

a Pytorch tensor with the same shape as `input` except the first dimension, whose length is that of `percentiles`

## Installation

Type `pip install -e .`

## Usage

Try out `python test.py` in the `examples` folder.
```
calling percentile on a tensor of dimension [10000    10    50]
numpy: 154.368ms,
CPU: 604.269ms, error: : 0.000002%.
GPU: 54.679ms, error: 0.000003%.
```
This implementation is worth it only on the GPU compared to the numpy version, but note that backward is implemented.
