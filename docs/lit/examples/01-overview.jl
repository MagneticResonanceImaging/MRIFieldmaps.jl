#=
# [MRIFieldmaps overview](@id 01-overview)

This page summarizes the Julia package
[`MRIFieldmaps`](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).
=#

#srcURL


#=
### Setup

Packages needed here.
Use `Pkg.add` as illustrated
[here](https://juliaimagerecon.github.io/Examples/generated/mri/1-nufft)
when using a package for the first time.
=#

using MRIFieldmaps: spdiff
using MIRTjim: jim, prompt; jim(:prompt, true)
using InteractiveUtils: versioninfo


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Overview

This package provides algorithms
for estimating fieldmaps in MRI.
Currently it supports B0 map estimation.
A future extension could support B1+ map estimation.

This page just discusses the regularization;
other examples illustrate specific fieldmap estimators.
=#


#=
## Regularization

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
pl = Array{Any}(undef, 2, 2)
for order in 1:2, d in 1:2
    sp = spdiff(dims; order)
    dif = reshape(sp[d] * vec(phantom), dims)
    pl[d,order] = jim(x, y, dif, "dim$d differences\norder=$order")
end
pp = jim(x, y, phantom, "Test image")
jim([[pp pp]; pl]...)


#=
## Support mask

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


include("../../../inc/reproduce.jl")
