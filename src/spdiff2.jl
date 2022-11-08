#=
spdiff2.jl
Sparse matrix for 2nd-order finite differences.
=#

using SparseArrays: spdiagm
using LinearAlgebra: I


"""
    spdiff2(n::Int; ending::Symbol = :remove, T::DataType = Int32)

Sparse matrix for 2nd-order finite differences.

# Option
- `T` element type, default 'Int32' to save memory
- `ending`
  - `:remove` (default) remove first and last rows, so `n-2 × n`
  - `:zero` keep first and last rows, akin to zero boundary conditions
  - `:first` use 1st-order finite differences for first and last rows
"""
function spdiff2(n::Int; ending::Symbol = :remove, T::DataType = Int32)
    tmp = spdiagm(-1 => -ones(T,n-1), +1 => -ones(T,n-1), 0 => fill(T(2), n))
    if ending === :remove
        return tmp[2:end-1,:]
    elseif ending === :zero
        return tmp
    elseif ending === :first
        tmp[1,1] = one(T)
        tmp[n,n] = one(T)
        return tmp
    else
        throw("bad option ending=$ending")
    end
end


#=
"""
    spdiff2(dims::Dims; kwargs...)

Return `Vector` of `length(dims)` sparse
2nd-order finite-difference matrices,
one for each dimension.
The `kwargs` are passed to `spdiff2` for each dimension.
Typically one will `vcat` these
to make a finite-difference matrix
suitable for the `vec` of a multi-dimensional array.

Examples:
- `spdiff2((4,5,6))[1] == kron(I(6*5), spdiff2(4))`
- `spdiff2((4,5,6))[2] == kron(I(6), spdiff2(5), I(4))`
- `spdiff2((4,5,6))[3] == kron(spdiff2(6), I(4*5))`
"""
function spdiff2(dims::Dims; kwargs...)
    diffs = spdiff2.(dims; kwargs...)
    I1 = i -> I(prod(dims[(i+1):end]))
    I2 = i -> I(prod(dims[1:(i-1)]))
    dfun = i -> kron(I1(i), diffs[i], I2(i))
    return dfun.(1:length(dims))
end
=#
