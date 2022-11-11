#=
b0scale.jl
based on mri_field_map_reg_scale.m
Matlab version originally by MJ Allison
2022-11-10 Jeff Fessler, Julia translation
=#

# todo: think about multiple coil case with smap

using Statistics: median

"""
    (yik, scalefactor) = b0scale(yik, etime; fmax, dmax) 

Scale images to account for R2* effects
and for magnitude variations
using `median(di)`
where
- `ri = sum_j sum_k |y_{ij} y_{ik}|^2 (t_k - t_j)^2`
- `di = ri / sum_k |y_{ik}|^2`

Only values where
`|yik| > fmax * maximum(abs, yik)` # todo
`di > dmax * maximum(di)`
affect `scalefactor`,
so it is fine to pass unmasked images here.

This scaling simplifies regularization parameter selection
for regularized fieldmap estimation.

# In
- `yik (dims..., nset)` scan images for `nset` different echo times
- `etime (nset)` echo times (units of sec if fieldmap is in Hz)

# Option
- `dmax::Real` threshold for relative `di` value (default `0.1`)

# Out
- `yik (dims..., nset)` scaled scan images
- `scalefactor = sqrt(median(rj))`
"""
function b0scale(
    yik::AbstractArray{<:Complex},
    etime::AbstractVector,
    ;
#   fmax::Real = 0.1,
    dmax::Real = 0.1,
)

    Base.require_one_based_indexing(yik, etime)

    dims = size(yik)
    yik = reshape(yik, :, dims[end]) # (*dims, nset)

    (nn, nset) = size(yik)

#=
    # Scale by median of first set of data to get rid of large mag_j effects.
    # (Not actually needed, but retained for consistency.)

    if fmax > 0
        y1 = abs.(yik[:,1])
        scalefactor = median(filter(>(fmax * maximum(y1)), y1))
        scalefactor == 0 && throw("median is zero?")
        yik ./= scalefactor
    else
        scalefactor = 1
    end
=#

    # Try to compensate for R2 effects on effective regularization.

    d = zeros(Float32, nn)
    for j in 1:nset
        for k in 1:nset
            d += abs2.(yik[:,j] .* yik[:,k]) * (etime[k] - etime[j])^2
        end
    end

    # divide by numerator of wj^mn -> sum(abs(y)^2)
    # todo: cite eqn #
    div0 = (x::Number, y::Number) -> iszero(y) ? 0 : x/y
    d = div0.(d, sum(abs2, yik; dims=2))

    # compute typical d value
    dtypical = median(filter(>(dmax * maximum(d)), d))

    # uniformly scale by the square root of dtypical
    scalefactor = sqrt(dtypical)
    yik ./= scalefactor

    yik = reshape(yik, dims) # (dims..., nset)

    return yik, scalefactor
end
