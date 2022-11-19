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

If `df` is nonempty (which always holds for water-fat case),
then perform discrete maximum-likelihood estimation using `fdict`.

# In
- `ydata (dims..., nc, ne)` `ne` sets of complex images for `nc ≥ 1` coils
- `echotime (ne)` vector of `ne ≥ 2` echo times (only first 2 are used)

# Options
- `smap (dims..., nc)` complex coil maps; default ones
- `threshold` set `finit` values where `|y1| < threshold * max(|y1|)`
   to the mean of the "good" values where `|y1| ≥ threshold * max(|y1|)`.
   default: `0`

# Options for water-fat case:
- `df` Δf values in water-fat imaging (def: `[]`) units Hz, e.g., `[440]` at 3T
- `relamp` relative amplitudes in multi-species water-fat (def: `[]`)
- `fband` frequency bandwith for `fdict; default 1 / minimum(echo time spacing)
- `nf` number of discrete frequencies to try; default 101
- `fdict` "dictionary" of discrete frequency values to try; default `LinRange(-1/2,1/2,nf) * fband`

# Out
- `finit` initial B0 fieldmap estimate in Hz
"""
function b0init(
    ydata::AbstractArray{<:Complex,D},
    echotime::Echotime{Te},
    ;
    smap::AbstractArray{<:Complex} = ones(ComplexF32, size(ydata)[1:end-1]),
    threshold::Real = 0,
    df::AbstractVector{<:RealU} = Float32[],
    relamp::AbstractVector{<:RealU} = ones(Float32, size(df)) / max(1, length(df)),
    T::DataType = eltype(1 / oneunit(Te)),
    kwargs...
)::Array{T,D-2} where {D, Te <: RealU}

    Base.require_one_based_indexing(echotime, smap, ydata)

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
    if !isempty(df)
        finit = b0init(reshape(yc,:,ne), echotime, df, relamp; kwargs...) # water-fat version
        finit = reshape(finit, dims)
    else
        y1 = selectdim(yc, ndim+1, 1)
        y2 = selectdim(yc, ndim+1, 2)
        finit = angle.(y2 .* conj(y1)) / Float32((echotime[2] - echotime[1]) * 2π)
    end

    # set background pixels to mean of "good" pixels.
    if threshold > 0
        mag1 = abs.(y1)
        good = mag1 .> (threshold * maximum(mag1))
        finit[.!good] .= sum(finit[good]) / count(good) # mean
    end

    return finit
end


# discrete maximum-likelihood approach
function b0init(
    ydata::AbstractMatrix{<:Complex}, # coil combined (*dim,ne)
    echotime::Echotime,
    df::AbstractVector{<:RealU},
    relamp::AbstractVector{<:RealU},
    ;
    fband::RealU = 1 / minimum(diff(sort(echotime))),
    nf::Int = 101,
    fdict::AbstractVector{<:RealU} = Float32.(LinRange(-0.5,0.5,nf) * fband),
)

    Base.require_one_based_indexing(df, echotime, fdict, relamp, ydata)

    (np,ne) = size(ydata)
    ne == length(echotime) || throw("ne mismatch $ne $(length(echotime))")
    # calculate the magnitude and angles used in the data-fit curvatures
    angy = angle.(ydata)

    # γ in eqn (5) of Lin&Fessler, of size (ne,2)
    γwf = [ones(ne) cis.(2f0π*echotime*df') * relamp]
    # Gamma in eqn (4) of Lin&Fessler, of size (ne,ne)
    if iszero(df)
        Gamma = ones(Float32, ne, ne) / ne
    else
        Gamma = γwf * inv(γwf'*γwf) * γwf'
    end

    set = 0
    nset = cumsum(1:ne-1)[end]
    wj_mag = zeros(Float32, np, nset)
    d2 = zeros(eltype(echotime), nset)
    ang2 = zeros(Float32, np, nset)

    for j in 1:ne, i in 1:ne # for each pair of echo times
        i ≥ j && continue # only need one of each pair of differences
        set += 1
        d2[set] = echotime[i] - echotime[j]
        @. wj_mag[:,set] = abs(ydata[:,i] * ydata[:,j])
        # difference of the echo times and angles
        ang2[:,set] = angy[:,j] - angy[:,i]
        wj_mag[:,set] .*= abs(Gamma[i,j])
        ang2[:,set] .+= angle(Gamma[i,j])
    end

    # try all discrete fdict values and find maximum likelihood
    ωdict = Float32.(2f0π * fdict)
    function maxlike(ip)
        wj = @view wj_mag[ip,:] # (nset)
        ϕj = @view ang2[ip,:] # (nset)
        sm = ωdict * d2' .+ ϕj' # (nd,nset)
        like = cos.(sm) * wj
        return fdict[argmax(like)]
    end

    return maxlike.(1:np)
end
