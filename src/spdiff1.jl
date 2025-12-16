#=
spdiff1.jl
Sparse matrix for 1st-order finite differences.
=#

using SparseArrays: spdiagm
using LinearAlgebra: I


"""
    spdiff1(n::Int; ending::Symbol = :remove, T::DataType = Int32)

Sparse `n × n` matrix
for 1st-order finite differences.

# Option
- `T` element type, default `Int32` to save memory
- `ending`
  - `:remove` (default) remove first difference
  - `:zero` keep first row, akin to zero boundary conditions
"""
function spdiff1(n::Int; ending::Symbol = :remove, T::DataType = Int32)
    sp = spdiagm(-1 => -ones(T,n-1), 0 => fill(one(T), n))
    if ending === :remove
        sp[1,1:2] .= zero(T)
    elseif !(ending === :zero)
        throw("bad option ending=$ending")
    end
    return sp
end
