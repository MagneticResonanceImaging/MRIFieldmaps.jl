#=
combine.jl
Complex coil combination
=#

export coil_combine


"""
    zdata, sos = coil_combine(ydata [, smap])

For estimating a B0 field map
from complete multi-coil image data,
it suffices to first do complex coil combination,
while tracking the sum-of-squares `sos`
for proper weighting.
When sensitivity maps are provided,
often `sos` is all 1's and 0's.

Uses a phase contrast coil combination approach
(reference below)
when sensitivity maps are not provided.

# In
- `ydata (dims..., nc, ne)` `ne ≥ 1` sets of complex images for `nc ≥ 1` coils
- `smap (dims..., nc)` complex coil maps (optional)

# Out
- `zdata (dims..., ne)` complex coil combination: `sum_c smap[c]' * ydata[c] ./ sos`
  or `sum_c ydata[c,1]' * ydata[c] ./ sos`
  (if `smap` not provided or `isnothing(smap)`)
- `sos (dims...)` sum-of-squares: `sum_c |smap[c]|^2`
  or `sqrt.(sum_c |ydata[c,1]|^2`
  (if `smap` not provided or `isnothing(smap)`)

See equation [13] in
M A Bernstein et al.,
"Reconstructions of Phase Contrast, Phased Array Multicoil Data", MRM 1994.
https://doi.org/10.1002/mrm.1910320308
"""
function coil_combine(
    ydata::AbstractArray{<:Complex},
    smap::AbstractArray{<:Complex},
)

    Base.require_one_based_indexing(smap, ydata)

    dims = size(ydata)[1:end-2]
    ndim = length(dims)
    nc = size(ydata)[ndim+1]
    ne = size(ydata)[ndim+2]

    # check dimensions
    size(smap) == (dims..., nc) ||
        throw("bad smap size $(size(smap)) != $((dims..., nc))")

    # coil combine image data
    sos = sum(abs2, smap, dims=ndim+1)
    sos = selectdim(sos, ndim+1, 1)
    smap = div0.(smap, sos)
    zdata = sum(conj(smap) .* ydata; dims=ndim+1) # coil combine
    zdata = selectdim(zdata, ndim+1, 1) # (dims..., ne)

    return zdata, sos
end


function coil_combine(
    ydata::AbstractArray{<:Complex},
    smap::Nothing = nothing,
)

    Base.require_one_based_indexing(ydata)

    dims = size(ydata)[1:end-2]
    ndim = length(dims)
    nc = size(ydata)[ndim+1]
    ne = size(ydata)[ndim+2]

    # The following is a dimensionality-agnostic way of writing
    # y = [[ydata[:,:,:,c,e] for c = 1:nc] for e = 1:ne]
    # (where the above assumes 3D data)
    y = ntuple(ne) do e
        ye = selectdim(ydata, ndim+2, e)
        [selectdim(ye, ndim+1, c) for c = 1:nc]
    end
    y1 = y[1]

    # Coil combine image data
    sos = sum(yc1 -> abs2.(yc1), y1)
    y1sos = map(yc1 -> yc1 ./ sos, y1)
    zdata = map(y) do ye
        # See eqn. [13] in the cited paper
        mapreduce(+, y1sos, ye) do yc1sos, yce
            conj.(yc1sos) .* yce
        end
    end
    zdata = cat(zdata..., dims=ndim+1)

    return zdata, sos
end
