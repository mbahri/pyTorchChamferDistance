# Chamfer Distance for pyTorch

This is a fork of chrdiller's implementation of the Chamfer Distance as a module for Pytorch. It is written as a custom C++/CUDA extension.

I modified it to also return the indices of the matching points, since it is common to need them as well.

As it is using pyTorch's [JIT compilation](https://pytorch.org/tutorials/advanced/cpp_extension.html), there are no additional prerequisite steps that have to be taken. Simply import the module as shown below; CUDA and C++ code will be compiled on the first run.

### Usage
```python
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are n_points x 3 matrices

dist1, dist2, idx1, idx2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(dist1)) + (torch.mean(dist2))


#...
```
