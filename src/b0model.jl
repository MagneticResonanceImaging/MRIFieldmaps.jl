# b0model.jl


"""
    b0model(fmap, xw, echotime; kwargs...)

Compute complex images for B0 field mapping.
This function is used mainly for simulation,
and for code testing.

Model:
`images[j,c,l] = smap[j,c] exp(ı 2π fmap[j] t_l) exp(-relax[j] t_l) x[j,l]`
where
`x[j,l] = xw[j] + xf[j] * sum_{p=0}^P α_p exp(ı 2π Δf_p t_l)`.

Field map estimation from multiple (`ne ≥ 2`) echo-time images,

# In
- `fmap (dims)` fieldmap (in Hz)
- `xw (dims)` water magnetization component
- `echotime (ne)` vector of `ne` echo time offsets (in sec)

# Options
- `smap (dims...[, nc])` complex coil maps, default `ones(size(fmap))`
- `xw (dims)` fat magnetization component, default `zeros(size(fmap))`
- `df` Δf values in water-fat imaging (def: `[0]`) units Hz, e.g., `[440]` at 3T
- `relamp` relative amplitudes in multi-peak water-fat (def: `[1]`)
- `relax (dims)` R2 or R2* relaxation in same units as `fmap`

# Out
- `ydata (dims..., [nc,] ne)` `ne` sets of complex images for `nc` coils
"""
function b0model(
    fmap::AbstractArray{Tf},
    xw::AbstractArray{Tx,D},
    echotime::Union{AbstractVector{<:RealU}, NTuple{N,<:RealU} where N},
    ;
    df::AbstractVector{<:RealU} = [0f0*oneunit(Tf)],
    relamp::AbstractVector{<:RealU} = [1f0],
    smap::AbstractArray{<:Complex} = ones(ComplexF32, size(fmap)..., 1),
    xf::AbstractArray{<:Complex} = zeros(Tx, size(fmap)),
    relax::AbstractArray{<:RealU} = zeros(Tf, size(fmap)),
)::Array{Tx,D+2} where {D, Tf <: RealU, Tx <: Complex}

    Base.require_one_based_indexing(echotime, fmap, smap, xf, xw)

    dims = size(fmap)
    ndim = length(dims)
    if ndims(smap) == ndim
        smap = reshape(smap, size(smap)..., 1) # nc=1 special case
    end
    nc = size(smap, ndim+1) # ≥ 1
    ne = length(echotime)

    # check dimensions
    dims == size(xw) || throw("bad xw size $(size(xw)) vs dims=$dims")
    dims == size(xf) || throw("bad xf size $(size(xf)) vs dims=$dims")
    (dims..., nc) == size(smap) ||
        throw("bad smap size $(size(smap)) vs dims=$dims & nc=$nc")

    # sum_{p=0}^P α_p exp(ı 2π Δf_p t_l)
    fat_phase = cis.(2f0π * echotime * df') * relamp
    # x[j,l] = xw[j] + xf[j] * sum_{p=0}^P α_p exp(ı 2π Δf_p t_l)
    fat_phase = reshape(fat_phase, ones(Int, ndim)..., ne)
    signal = @. xw + xf * fat_phase

    # images[j,c,l] = smap[j,c] exp(ı 2π fmap[j] t_l) exp(-relax[j] t_l) x[j,l]
    times = reshape(echotime, ones(Int, ndim)..., ne)
    relax = @. exp(-relax * times) # exp(-relax[j] t_l)
    phase = @. cis(2f0π * fmap * times) # exp(ı 2π fmap[j] t_l)

    images = @. phase * relax * signal
    images = reshape(images, dims..., 1, ne)
    images = smap .* images # (dims..., nc, ne)
    return images
end
