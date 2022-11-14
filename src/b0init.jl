#=
b0init.jl
Initialize b0 fieldmap using the classic "phase difference"
=#

using StatsBase: mean

export b0init


"""
    finit = b0init(ydata, echotime; kwargs...)

Classic B0 field map estimation
based on the phase difference
of complex images at two different echo times.
If sensitivity maps (`smap`) are provided,
complex coil combination is done first.
This code works with images of arbitrary dimensions (2D, 3D, etc.),
and with multiple coils.

In the usual case where `echotime` has units of seconds,
the returned B0 fieldmap will have units of Hz.

# In
- `ydata (dims..., nc, ne)` `ne` sets of complex images for `nc ≥ 1` coils
- `echotime (ne)` vector of `ne ≥ 2` echo time offsets (only first 2 are used)

# Options
- `smap (dims..., nc)` complex coil maps; default ones
- `threshold` set `finit` values where `|y1| < threshold * max(|y1|)`
   to the mean of the "good" values where `|y1| ≥ threshold * max(|y1|)`.
   default: `0`

# Out
- `finit` initial B0 fieldmap estimate in Hz
"""
function b0init(
    ydata::Array{<:Complex},
    echotime::Union{AbstractVector{<:RealU}, NTuple{N,<:RealU} where N},
    ;
    smap::AbstractArray{<:Complex} = ones(ComplexF32, size(ydata)[1:end-1]),
    threshold::Real = 0,
    kwargs...
)

    Base.require_one_based_indexing(echotime, smap)

    dims = size(ydata)[1:end-2]
    ndim = length(dims)
    nc = size(ydata)[ndim+1]
    ne = size(ydata)[ndim+2]

    # check dimensions
    size(smap) == (dims..., nc) ||
        throw("bad smap size $(size(smap)) != $((dims..., nc))")

    ne == length(echotime) || throw("need echotime to have length ne=$ne")
    ne ≥ 2 || throw("need ne=$ne ≥ 2")

    # coil combine image data
    yc = sum(conj(smap) .* ydata; dims=ndim+1) # coil combine
    yc = selectdim(yc, ndim+1, 1) # (dims..., ne)

    # initial fieldmap via phase difference of first two echo times
    y1 = selectdim(yc, ndim+1, 1)
    y2 = selectdim(yc, ndim+1, 2)
    finit = angle.(y2 .* conj(y1)) / Float32((echotime[2] - echotime[1]) * 2π)

    # set background pixels to mean of "good" pixels.
    if threshold > 0
        mag1 = abs.(y1)
        good = mag1 .> (threshold * maximum(mag1))
        finit[.!good] .= sum(finit[good]) / count(good) # mean
    end

    return finit
end
