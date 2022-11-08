#=
spdiff1.jl
Sparse matrix for 1st-order finite differences.
=#

using SparseArrays: spdiagm
using LinearAlgebra: I


"""
    spdiff1(n::Int; ending::Symbol = :remove, T::DataType = Int32)

Sparse matrix for 1st-order finite differences.

# Option
- `T` element type, default 'Int32' to save memory
- `ending`
  - `:remove` (default) remove first row, so `n-1 × n`
  - `:zero` keep first row, akin to zero boundary conditions
"""
function spdiff1(n::Int; ending::Symbol = :remove, T::DataType = Int32)
    tmp = spdiagm(-1 => -ones(T,n-1), 0 => fill(one(T), n))
    if ending === :remove
        return tmp[2:end,:]
    elseif ending === :zero
        return tmp
    else
        throw("bad option ending=$ending")
    end
end


#=
"""
    spdiff1(dims::Dims; kwargs...)

Return `Vector` of `length(dims)` sparse
1st-order finite-difference matrices,
one for each dimension.
The `kwargs` are passed to `spdiff1` for each dimension.
Typically one will `vcat` these
to make a finite-difference matrix
suitable for the `vec` of a multi-dimensional array.

Examples:
- `spdiff1((4,5,6))[1] == kron(I(6*5), spdiff1(4))`
- `spdiff1((4,5,6))[2] == kron(I(6), spdiff1(5), I(4))`
- `spdiff1((4,5,6))[3] == kron(spdiff1(6), I(4*5))`
"""
function spdiff1(dims::Dims; kwargs...)
    diffs = diff1.(dims; kwargs...)
    I1 = i -> I(prod(dims[(i+1):end]))
    I2 = i -> I(prod(dims[1:(i-1)]))
    dfun = i -> kron(I1(i), diffs[i], I2(i))
    return dfun.(1:length(dims))
end
=#
