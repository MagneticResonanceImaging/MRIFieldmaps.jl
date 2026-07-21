#=
spdiff2.jl
Sparse matrix for 2nd-order finite differences.
=#

using SparseArrays: spdiagm
using LinearAlgebra: I


"""
    spdiff2(n::Int; ending::Symbol = :remove, T::DataType = Int32)

Sparse `n × n` matrix
for 2nd-order finite differences.

# Option
- `T` element type, default `Int32` to save memory
- `ending`
  - `:remove` (default) remove first and last finite difference
  - `:zero` keep first and last rows, akin to zero boundary conditions
  - `:first` use 1st-order finite differences for first and last rows
"""
function spdiff2(n::Int; ending::Symbol = :remove, T::DataType = Int32)
    sp = spdiagm(-1 => -ones(T,n-1), +1 => -ones(T,n-1), 0 => fill(T(2), n))
    if ending === :remove
        sp[1,1:3] .= zero(T)
        sp[n,(n-2):n] .= zero(T)
    elseif ending === :first
        sp[1,1] = one(T)
        sp[n,n] = one(T)
    elseif !(ending === :zero)
        throw("bad option ending=$ending")
    end
    return sp
end
