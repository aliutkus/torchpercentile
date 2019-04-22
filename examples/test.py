import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torchpercentile import Percentile


if __name__ == "__main__":
    # defining the number of tests
    ntests = 2

    # problem dimensions
    data_shape = (10, 50)
    num_data = 10000

    percentiles = [10, 20, 50]#torch.linspace(0, 100, 10)

    # draw the data
    x = torch.rand(num_data, *data_shape)

    print('calling percentile on a tensor of dimension', torch.tensor(x.shape).numpy())

    # calling the numpy version
    xnp = x.detach().numpy()
    t0_np = time.time()
    percentiles_np = np.percentile(xnp, percentiles, axis=0)
    t1_np = time.time()
    percentiles_np = torch.tensor(percentiles_np).float()

    display_str = 'numpy: %0.3fms, \n' % ((t1_np-t0_np)*1000)

    # calling the CPU version
    t0_cpu = time.time()
    percentiles_cpu = Percentile()(x, percentiles)
    t1_cpu = time.time()
    error = torch.norm(
        percentiles_cpu - percentiles_np)/torch.norm(percentiles_np)*100.

    display_str += 'CPU: %0.3fms, error: : %f%%.\n' % (
        (t1_cpu-t0_cpu)*1000, error)

    if torch.cuda.is_available():
        x = x.to('cuda')

        # launching the cuda version
        t0_gpu = time.time()
        percentiles_gpu = Percentile()(x, percentiles)
        t1_gpu = time.time()

        # compute the difference between both
        error = torch.norm(
            percentiles_np - percentiles_gpu.to('cpu')
            )/torch.norm(percentiles_np)*100.

        display_str += 'GPU: %0.3fms, error: %f%%.' % (
            (t1_gpu-t0_gpu)*1000, error)

    print(display_str)
