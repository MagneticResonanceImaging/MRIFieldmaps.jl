#---------------------------------------------------------
# # [MRIfieldmap overview](@id 01-overview)
#---------------------------------------------------------

#=
This page summarizes the Julia package
[`MRIfieldmap`](https://github.com/JeffFessler/MRIfieldmap.jl).

This page was generated from a single Julia file:
[01-overview.jl](@__REPO_ROOT_URL__/01-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`01-overview.ipynb`](@__NBVIEWER_ROOT_URL__/01-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`01-overview.ipynb`](@__BINDER_ROOT_URL__/01-overview.ipynb).


# ### Setup

# Packages needed here.

using MRIfieldmap: spdiff
using MIRTjim: jim, prompt; jim(:prompt, true)
using InteractiveUtils: versioninfo


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
### Overview

This package provides algorithms
for estimating fieldmaps in MRI.

Currently it supports B0 map estimation.

A future extension could support B1+ map estimation.
=#


#=
### Regularization
Some methods in this package use sparse matrices
to perform finite-difference operations.
Those operations could be performed
with a `LinearMap` based on `diff`,
but the fastest algorithms here
use preconditioning based on
[incomplete Cholesky factorization](https://en.wikipedia.org/wiki/Incomplete_Cholesky_factorization)
of a Hessian matrix,
and such factorization
is supported for sparse matrices,
but probably not for general linear maps.

It may be helpful
to visualize
these sparse matrices.

Here are 1D 1st-order and 2nd-order finite differences.
=#

D1 = spdiff((8,); order=1)[1]
D2 = spdiff((8,); order=2)[1]
clim = (-1, 2)
color = :cividis
jim(
 jim(Matrix(D1)', "1st-order"; color, clim, prompt=false),
 jim(Matrix(D2)', "2nd-order"; color, clim, prompt=false),
)

# There are other boundary conditions available:
D1z = spdiff((8,); order=1, ending=:zero)[1]
D2z = spdiff((8,); order=2, ending=:zero)[1]
D2f = spdiff((8,); order=2, ending=:first)[1]
jim(
 jim(Matrix(D1z)', "1st-order, zero ends"; color, clim, prompt=false),
 jim(Matrix(D2z)', "2nd-order, zero ends"; color, clim, prompt=false),
 jim(Matrix(D2f)', "2nd-order, 1st ends"; color, clim, prompt=false),
 ; layout = (3,1),
)

#=
For multi-dimensional arrays,
regularizers need finite-differences along each dimension,
and these are constructed with Kronecker products
and are applied to the `vec` of an array.
=#

dims = (9,8)
d1 = spdiff(dims; order=1)
d2 = spdiff(dims; order=2)
jim(
 jim(Matrix(d1[1])', "1st-order, 1st dim"; color, clim, prompt=false),
 jim(Matrix(d1[2])', "1st-order, 2nd dim"; color, clim, prompt=false),
 jim(Matrix(d2[1])', "2nd-order, 1st dim"; color, clim, prompt=false),
 jim(Matrix(d2[2])', "2nd-order, 2nd dim"; color, clim, prompt=false),
)


#=
Here is an illustration of applying these finite-difference matrices
to a simple test phantom.
Note the use of `vec` and `reshape` for display.
=#
dims = (40,30)
x = LinRange(-1, 1, dims[1])
y = LinRange(-1, 1, dims[2])
phantom = @. abs(x) + abs(y') < 0.5
sp = spdiff(dims; order=1)
d1 = reshape(sp[1] * vec(phantom), dims)
d2 = reshape(sp[2] * vec(phantom), dims)
jim(
 jim(x, y, phantom, "Test image"),
 jim(x, y, d1, "1st dim differences"),
 jim(x, y, d2, "2nd dim differences"),
)


#=
### Support mask

Often we want to estimate a fieldmap
over some spatial support "mask"
that is smaller than the entire image,
e.g.,
only in voxels where the signal is sufficiently large.
See the
[ImageGeoms.jl](https://juliaimagerecon.github.io/ImageGeoms.jl/stable/generated/examples/2-mask)
documentation
about the related `embed` and `maskit` operations.
=#


#=
### Reproducibility
This page was generated with the following version of Julia:
=#

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
