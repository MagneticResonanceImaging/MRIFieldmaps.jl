#=
b0scale.jl
based on mri_field_map_reg_scale.m
Matlab version originally by MJ Allison
2022-11-10 Jeff Fessler, Julia translation
=#

# todo: think about multiple coil case with smap

using StatsBase: median

"""
    (ydata, scalefactor) = b0scale(ydata, echotime; fmax, dmax)

Scale complex images `ydata` to account for R2* effects
and for magnitude variations
using `median(di)`
where
- `ri = sum_j sum_k |y_{ij} y_{ik}|^2 (t_k - t_j)^2`
- `di = ri / sum_k |y_{ik}|^2`

Only values where
`|ydata| > fmax * maximum(abs, ydata)` # todo
`di > dmax * maximum(di)`
affect `scalefactor`,
so it is fine to pass unmasked images here.

This scaling simplifies regularization parameter selection
for regularized fieldmap estimation.

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
#   fmax::Real = 0.1,
    dmax::Real = 0.1,
) where Te <: RealU

    Base.require_one_based_indexing(ydata, echotime)

    dims = size(ydata)
    ydata = reshape(ydata, :, dims[end]) # (*dims, ne)

    (nn, ne) = size(ydata)
    echotime = echotime / oneunit(eltype(echotime)) # units are irrelevant here

#=
    # Scale by median of first set of data to get rid of large mag_j effects.
    # (Not actually needed, but retained for consistency.)

    if fmax > 0
        y1 = abs.(ydata[:,1])
        scalefactor = median(filter(>(fmax * maximum(y1)), y1))
        scalefactor == 0 && throw("median is zero?")
        ydata ./= scalefactor
    else
        scalefactor = 1
    end
=#

    # Try to compensate for R2 effects on effective regularization.

    d = reduce(+,
        abs2.(ydata[:,j] .* ydata[:,k]) * (echotime[k] - echotime[j])^2
        for j in 1:ne, k in 1:ne
    )

    # divide by numerator of wj^mn -> sum(abs(y)^2)
    # todo: cite eqn #
    div0 = (x::Number, y::Number) -> iszero(y) ? 0 : x/y # todo: zero(Tx/Ty)
    d = div0.(d, sum(abs2, ydata; dims=2))

    # compute typical d value
    dtypical = median(filter(>(dmax * maximum(d)), d))

    # uniformly scale by the square root of dtypical
    scalefactor = sqrt(dtypical)
    ydata ./= scalefactor

    ydata = reshape(ydata, dims) # (dims..., ne)

    return ydata, scalefactor
end
