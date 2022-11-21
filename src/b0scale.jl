#=
b0scale.jl
based on mri_field_map_reg_scale.m
Matlab version originally by MJ Allison
2022-11-10 Jeff Fessler, Julia translation
=#

using StatsBase: median

function div0(x::Tx, y::Ty) where {Tx <: Number, Ty <: Number}
    return iszero(y) ? zero(oneunit(Tx)/oneunit(Ty)) : x/y
end


"""
    (ydata, scalefactor) = b0scale(ydata, echotime; dmax)

Scale complex images `ydata` to account for R2* effects
and for magnitude variations
using `median(di)`
where
- `ri = sum_j sum_k |y_{ij} y_{ik}|^2 (t_k - t_j)^2`
- `di = ri / sum_k |y_{ik}|^2`

Only values where
`di > dmax * maximum(di)`
affect `scalefactor`,
so it is fine to pass unmasked images here.

This normalization simplifies regularization parameter selection
for regularized B0 fieldmap estimation.
See eqn (9) and (15) of Funai & Fessler, Oct. 2008, IEEE T-MI,
http://doi.org/10.1109/TMI.2008.923956

# In
- `ydata (dims..., ne)` scan images for `ne` different echo times
- `echotime (ne)` echo times (units of sec if fieldmap is in Hz)

# Option
- `dmax::Real` threshold for relative `di` value (default `0.1`)

# Out
- `ydata (dims..., ne)` scaled scan images
- `scalefactor = sqrt(median(rj))`
"""
function b0scale(
    ydata::AbstractArray{<:Complex},
    echotime::Union{AbstractVector{Te}, NTuple{N,Te} where N},
    ;
    dmax::Real = 0.1,
) where Te <: RealU

    Base.require_one_based_indexing(ydata, echotime)

    dims = size(ydata)
    ydata = reshape(ydata, :, dims[end]) # (*dims, ne)

    (nn, ne) = size(ydata)
    echotime = echotime / oneunit(eltype(echotime)) # units are irrelevant here

    # Try to compensate for R2 effects on effective regularization.
    # This uses eqn (15) of Funai&Fessler, with the numerator of (9):
    d = reduce(+,
        abs2.(ydata[:,j] .* ydata[:,k]) * (echotime[k] - echotime[j])^2
        for j in 1:ne, k in 1:ne
    )

    # Divide by denominator of eqn (9) of Funai&Fessler
    d = div0.(d, sum(abs2, ydata; dims=2))

    # compute typical d value
    dtypical = median(filter(>(dmax * maximum(d)), d))

    # uniformly scale by the square root of dtypical
    scalefactor = sqrt(dtypical)
    ydata ./= scalefactor

    ydata = reshape(ydata, dims) # (dims..., ne)

    return ydata, scalefactor
end
