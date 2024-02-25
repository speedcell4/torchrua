# import torch
# from torch import Tensor
#
# from torchrua import major_sizes_to_ptr
#
#
# @torch.no_grad()
# def major_sizes_to_span_ptr(sizes: Tensor):
#     major_ptr, minor_sizes = major_sizes_to_ptr(sizes=sizes)
#     x_ptr, z_ptr = major_sizes_to_ptr(sizes=minor_sizes + 1)
#
#     return major_ptr[z_ptr], minor_sizes[z_ptr], x_ptr, minor_sizes + 1
