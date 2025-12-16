#=
spdiff.jl
Collection of sparse matrices for finite differences.
=#

using SparseArrays: spdiagm
using LinearAlgebra: I

export spdiff


"""
    spdiff(dims::Dims; order=1, kwargs...)

Return `Vector` of `length(dims)`
sparse finite-difference matrices
of order `order`,
one for each dimension.

Typically one will `vcat` the vector output
to make a sparse finite-difference matrix
suitable for the `vec` of a multi-dimensional array.

The `kwargs` are passed
to `spdiff1` (for `order = 1`)
or `spdiff2` (for `order = 2`).
These functions are called once for each dimension.
The options are `ending` and `T`.

Examples:
- `spdiff((4,5,6))[1] == kron(I(6*5), spdiff1(4))`
- `spdiff((4,5,6); order=1)[2] == kron(I(6), spdiff1(5), I(4))`
- `spdiff((4,5,6); order=2)[3] == kron(spdiff2(6), I(4*5))`
"""
function spdiff(dims::Dims; order::Int = 1, kwargs...)
    diffs =
        order == 1 ? spdiff1.(dims; kwargs...) :
        order == 2 ? spdiff2.(dims; kwargs...) :
        throw("unsupported order $order")
    I1 = i -> I(prod(dims[(i+1):end]))
    I2 = i -> I(prod(dims[1:(i-1)]))
    dfun = i -> kron(I1(i), diffs[i], I2(i))
    return dfun.(1:length(dims))
end
