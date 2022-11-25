# b0map.jl

using LinearAlgebra: norm, cholesky
using ImageGeoms: embed
using SparseArrays: spdiagm, diag
using LimitedLDLFactorizations: lldl
#using MRIFieldmaps: spdiff, b0init, coil_combine

export b0map


# This function is isolated to facilitate code coverage
function _check_descent(ddir, grad, npregrad, oldinprod::Number, warned_dir::Bool)
    # check if correct descent direction
    if sign(ddir' * grad) > 0
        if !warned_dir
            warned_dir = true
            @warn "wrong direction so resetting"
            @warn "<ddir,grad>=$(ddir'*grad), |ddir|=$(norm(ddir)), |grad|=$(norm(grad))"
        end
        # reset direction if not descending
        ddir = npregrad
        oldinprod = 0
    end
    return ddir, oldinprod, warned_dir
end


"""
    (fhat, times, out) = b0map(ydata, echotime; kwargs...)

Field map estimation from multiple (`ne ≥ 2`) echo-time images,
using preconditioned nonlinear CG (NCG) with a monotonic line search.
This code works with images of arbitrary dimensions (2D, 3D, etc.),
and with multiple coils.

Caution:
single coil data must be reshaped to size `(dims..., 1, ne)`.

The cost function for the single-coil case is:
`cost(w) = ∑_{j=1}^{#voxel} ∑_{m=1}^ne ∑_{n=1}^ne
    |y_{mj} y_{nj}| wj (1 - cos(w_j * (t_m - t_n) + ∠y_{ni} - ∠y_{mj})) + R(w)`
where `t_n` denotes the echo time of the `n`th scan and
`R(w) = 0.5 * | C * w |^2`
is a quadratic roughness regularizer
based on 1st-order or 2nd-order finite differences.
See the documentation for the general multi-coil case.

The initial field map `finit` and output `fhat` are field maps in Hz,
but internally the code works with `ω = 2π f` (rad/s).

# In
- `ydata (dims..., nc, ne)` `ne` sets of complex images for `nc ≥ 1` coils
- `echotime::Echotime (ne ≥ 2)` echo time offsets (in seconds)

# Options
- `finit (dims)` initial fieldmap estimate (in Hz); default from `b0init()`
- `smap (dims..., nc)` complex coil maps; default `ones(size(ydata)[1:end-1])`
- `mask (dims...)` logical reconstruction mask; default: `trues(size(finit))`
- `ninner` # of inner iterations for monotonic line search
  default: `3` inner iterations
- `b0init_args::NamedTuple = (;)` options for `b0init`, such as `threshold`
- `niter` # of outer iterations (def: `30`)
- `order` order of the finite-difference matrix (def: `2`)
- `l2b` `log2` of regularization parameter (def: `-6`)
- `gamma_type` CG direction:
  * `:PR` = Polak-Ribiere (default)
  * `:FR` = Fletcher-Reeves
- `precon` Preconditioner:
  * `:I` (nothing)
  * `:diag`
  * `:chol` may require too much memory
  * `:ichol` (default)
- `reset` # of iterations before resetting direction (def: `Inf`)
- `df` Δf values in water-fat imaging (def: `[0]`) units Hz, e.g., `[440]` at 3T
- `relamp` relative amplitudes in multi-peak water-fat (def: `[1]`)
- `lldl_args::NamedTuple` options for `lldl`, default: `(;memory=2)`
- `track::Bool` if `true` then track cost and save all iterations (def: `false`)
- `chat::Bool = true` # `@info` updates each iteration
- `chat_iter::Int = 10` # print progress report every few iterations.

# Out
- `fhat` final fieldmap estimate in Hz
- `times (niter+1)` wall time for each iteration
- `out::NamedTuple` that contains:
   * `(xw, xf) (dims)` water / fat images if `!iszero(df)`
   * `finit (dims)` initial fieldmap
- if `track == true` then `out` also contains:
   * `costs (niter+1)` (nonconvex) cost for each iteration
   * `fhats (dims, niter+1)` fieldmap estimates every iteration

The algorithm is based on the paper:
C Y Lin, J A Fessler,
"Efficient Regularized Field Map Estimation in 3D MRI", IEEE TCI 2020
http://doi.org/10.1109/TCI.2020.3031082
http://arxiv.org/abs/2005.08661

Please cite that paper if you use this code.
"""
b0map


function b0map(
    ydata::Array{<:Complex, D},
    echotime::Echotime{Te},
    ;
    smap::AbstractArray{<:Complex} = ones(ComplexF32, size(ydata)[1:end-1]),
    df::AbstractVector{<:RealU} = eltype(1/oneunit(Te))[],
    relamp::AbstractVector{<:RealU} = ones(Float32, size(df)) / max(1, length(df)),
    b0init_args::NamedTuple = (;),
    finit::AbstractArray{<:RealU} = b0init(ydata, echotime; smap, df, relamp, b0init_args...),
    kwargs...
) where {D, Te <: RealU}

    D < 4 && @warn("D = $D < 4 is appropriate only for 1D MRI")

    (fhat, times, out) = b0map(finit, ydata, echotime; smap, df, relamp, kwargs...)
    out = merge(out, (; finit))
    return (fhat, times, out)
end


function b0map(
    finit::AbstractArray{<:RealU},
    ydata::Array{<:Complex},
    echotime::Echotime,
    ;
    smap::AbstractArray{<:Complex} = ones(ComplexF32, size(finit)..., 1),
    mask::AbstractArray{<:Bool} = trues(size(finit)),
    kwargs...
)

    Base.require_one_based_indexing(echotime, finit, mask, smap)

    dims = size(finit)
    nc = size(smap)[end] # ≥ 1
    ne = length(echotime)

    # check dimensions
    (dims..., nc, ne) == size(ydata) || throw("bad ydata size")
    dims == size(mask) || throw("bad mask size $(size(mask)) vs dims=$dims")
    (dims..., nc) == size(smap) ||
        throw("bad smap size $(size(smap)) vs dims=$dims & nc=$nc")

    zdata, sos = coil_combine(ydata, smap) # coil combine image data

    (fhat, times, out) = b0map(
        finit[mask],
        reshape(zdata, :, ne)[vec(mask), :],
        sos[mask],
        echotime,
        mask ;
        kwargs...
    )

    if hasfield(typeof(out), :fhats)
        out = merge(out, (;fhats = embed(out.fhats, mask)))
    end
    if hasfield(typeof(out), :xw)
        xw = embed(out.xw, mask)
        xf = embed(out.xf, mask)
        out = merge(out, (; xw, xf))
    end
    return embed(fhat, mask), times, out
end


"""
    b0map()

This version expects masked column-like inputs.
For expert use only.

# In
- `finit (np)` initial estimate in Hz (`np` is # of pixels in mask)
- `zdata (np, ne)` `ne` sets of coil-combined measurements
- `sos (np)` sum-of-squares of coil maps
- `echotime (ne)` vector of `ne` echo time offsets
- `mask (N)` logical reconstruction mask
"""
function b0map(
    finit::AbstractVector{<:RealU},
    zdata::AbstractMatrix{<:Complex},
    sos::AbstractVector{<:Real},
    echotime::Echotime,
    mask::AbstractArray{<:Bool},
    ;
    ninner::Int = 3, # subiterations
    niter::Int = 30,
    order::Int = 2,
    l2b::Real = -6,
    gamma_type::Symbol = :PR,
    precon::Symbol = :ichol,
    reset::Real = Inf,
    df::AbstractVector{<:RealU} = eltype(finit)[],
    relamp::AbstractVector{<:RealU} = ones(Float32, size(df)) / max(1, length(df)),
    chat::Bool = true,
    chat_iter::Int = 10, # progress report this often
    track::Bool = false,
    lldl_args::NamedTuple = (; memory = 2),
)

    Base.require_one_based_indexing(df, echotime, finit, mask, relamp, sos, zdata)
    t0 = time() # start timer

    units = oneunit(eltype(finit)) # could be 1Hz or 1
    if precon === :ichol || # lldl does not support units :(
        precon === :chol # https://github.com/JuliaLang/julia/issues/47655
        finit = finit / units
        df = df / units
        echotime = echotime * units
        fixunits = units
    else
        fixunits = 1
    end

    # check dimensions
    (np, ne) = size(zdata)
    ne == length(echotime) || throw("need echotime to have length ne=$ne")
    np == length(finit) || throw("need finit to have length np=$np")
    np == length(sos) || throw("need sos to have length np=$np")
    count(mask) == np || throw("bad mask count")
    length(relamp) == length(df) ||
        throw("inconsistent length df $(length(df)) & relamp $(length(relamp))")

    # calculate the magnitude and angles used in the data-fit curvatures
    angz = angle.(zdata)
    if !iszero(df)
        # γ in eqn (5) of Lin&Fessler, of size (ne,2)
        γwf = [ones(ne) cis.(2f0π*echotime*df') * relamp]
        # Gamma in eqn (4) of Lin&Fessler, of size (ne,ne)
        # @show cond(γwf)
        Gamma = γwf * inv(γwf'*γwf) * γwf'
    end

    # Precompute data-dependent magnitude and phase factors.
    nset = ne * (ne-1) ÷ 2 # "half" of ne × ne double sum
    wj_mag = zeros(Float32, np, nset)
    d2 = zeros(eltype(Float32(echotime[1])), 1, nset) # trick
    ang2 = zeros(Float32, np, nset)
    set = 0
    for j in 2:ne, i in 1:(j-1) # for each pair of scans: "m,n" in (3) in Lin&Fessler
        set += 1
        d2[set] = echotime[i] - echotime[j]
        @. wj_mag[:,set] = abs(zdata[:,i] * zdata[:,j])
        # difference of the echo times and angles
        @. ang2[:,set] = angz[:,j] - angz[:,i]
        # Γ effect:
        if iszero(df) # no fat
             wj_mag[:,set] ./= ne # eqn. (4) in paper for non-fat case
        else
            wj_mag[:,set] .*= abs(Gamma[i,j])
            ang2[:,set] .+= angle(Gamma[i,j])
        end
    end

    # apply s_j^2 and (t_l - t_l')
    wj_mag .*= sos
    wm_deltaD = wj_mag .* d2 # for derivative
    wm_deltaD2 = wj_mag .* abs2.(d2) # for curvature bound


    # sparse finite-difference regularization matrix (?,np)
    C = vcat(spdiff(size(mask); order)...)

    # remove all rows of C that involve non-mask pixels to avoid "leaking"
    good = iszero.(abs.(C) * .!vec(mask))
    C = C[good,:]
    C = C[:,vec(mask)] # (?,np) apply mask
    β = 2f0^l2b / oneunit(eltype(finit))^2 # unit balancing!
    C = sqrt(β) * C
    CC = C' * C

    # prepare output variables
    times = zeros(niter+1)
    if track
        out_fs = zeros(eltype(finit), length(finit), niter+1)
        out_fs[:,1] .= finit
        costs = zeros(niter+1)

#=
        sm = w * d2' .+ ang2 # (np, ne)
        cost0d = sum(@. wj_mag * (1 - cos(sm)))
        cost0r = 0.5 * norm(C*w)^2
        cost0 = cost0d + cost0r
=#
    end

    if precon === :diag
        dCC = diag(CC)
        T = eltype(Float32(oneunit(eltype(dCC))))
        dCC = Vector{T}(dCC)
    end

    # initialize NCG variables
    w = 2f0π * finit
    CCw = CC * w
    oldinprod = 0
    warned_dir = false
    warned_step = false
    ngradO = nothing
    ddir = nothing
    gamma = nothing

    # begin outer iterations
    chat && @info "ite_solve: NCG-MLS with precon $precon"

    for iter in 1:niter
        # compute the gradient of the cost function and curvatures
        (hderiv, hcurv, sm) = Adercurv(d2, ang2, wm_deltaD, wm_deltaD2, w)

        grad = hderiv + CCw
        ngrad = -grad

        if track
            costs[iter] = sum(@. wj_mag * (1 - cos(sm))) + 0.5 * norm(C*w)^2
            chat && @info "iter $(iter-1), cost $(costs[iter])"
        end

        # apply preconditioner

        if precon === :I
            npregrad = ngrad * units^2 # note: crucial for unit balance!

        elseif precon === :diag
            Hdiag = hcurv + dCC
            npregrad = ngrad ./ Hdiag

        elseif precon === :chol
            H = spdiagm(hcurv) + CC
            PC = cholesky(H) # may run out of memory for large problems
            npregrad = Vector{Float32}(PC \ ngrad)
            # preceding cast is due to CHOLMOD limitation as of 2022-11
            # https://github.com/JuliaSparse/SparseArrays.jl/issues/111

        elseif precon === :ichol
            H = spdiagm(hcurv) + CC
            # todo: reinterpret for units
            PI = lldl(H; lldl_args...)
            npregrad = PI \ ngrad

        else
            throw("unknown precon $precon")
        end

        # compute CG direction
        newinprod = ngrad' * npregrad

        if oldinprod == 0 || mod(iter, reset) == 0
            ddir = npregrad
            gamma = 0
            ngradO = ngrad
        else
            if gamma_type === :FR # Fletcher-Reeves
                gamma = newinprod / oldinprod

            elseif gamma_type === :PR # Polack-Ribeir
                gamma = real((ngrad - ngradO)' * npregrad) / oldinprod
                ngradO = ngrad

                if (gamma < 0)
                    @info "RESETTING GAMMA, iter=$iter"
                    gamma = 0
                end

            else
                throw("bad gamma_type $gamma_type")
            end

            ddir = npregrad + gamma * ddir
        end
        oldinprod = newinprod

        ddir, oldinprod, warned_dir =
            _check_descent(ddir, grad, npregrad, oldinprod, warned_dir)
#=
        # check if correct descent direction
        if sign(ddir' * grad) > 0
            if !warned_dir
                warned_dir = true
                @warn "wrong direction so resetting"
                @warn "<ddir,grad>=$(ddir'*grad), |ddir|=$(norm(ddir)), |grad|=$(norm(grad))"
            end
            # reset direction if not descending
            ddir = npregrad
            oldinprod = 0
        end
=#

        # step size in search direction
        Cdir = C * ddir # caution: can be a big array for 3D problems

        # compute the monotonic line search using quadratic surrogates
        CdCd = Cdir' * Cdir
        CdCw = ddir' * CCw
        step = 0
        for is in 1:ninner

            # compute the curvature and derivative for subsequent steps
            if !iszero(step)
                tmp = w + step * ddir
                (hderiv,hcurv) = Adercurv(d2, ang2, wm_deltaD, wm_deltaD2, tmp)
            end

            # compute numer and denom of the Huber's algorithm based line search
            denom = abs2.(ddir)' * hcurv + CdCd
            numer = ddir' * hderiv + (CdCw + step * CdCd)

            step = iszero(denom) ? step0 : step - numer / denom # update line search
            iszero(step) && @warn "found exact solution??? step=0 now!?"
        end

        # update the estimate and the finite differences of the estimate
        CCw += step * (C' * Cdir)
        w += step * ddir

        # save any iterations that are required (with times)
        times[iter+1] = time() - t0
        if track
            out_fs[:,iter+1] = w / 2f0π
        end
        # display counter
        if mod1(iter, chat_iter) == 1
            @info "$iter of $niter"
        end
    end

    if track
        sm = w * vec(d2)' .+ ang2
    #   (_, _, sm) = Adercurv(d2, ang2, wm_deltaD, wm_deltaD2, w)
        costs[niter+1] = sum(@. wj_mag * (1 - cos(sm))) + 0.5 * norm(C*w)^2
    #   chat && @info ' ite: %d , cost: %f3\n', iter, cost(iter+1))
        out = (fhats = out_fs * fixunits, costs)
    else
        out = (; )
    end

    # output water & fat images
    if !iszero(df)
        x = decomp(w, relamp, echotime, zdata, γwf)
        out = merge(out, (xw = x[1,:], xf = x[2,:]))
    end

    return w * (fixunits / 2f0π), times, out # return Hz
end


# compute the data-fit derivatives and curvatures as in Funai 2008 paper
function Adercurv(d2, ang2, wm_deltaD, wm_deltaD2, w)
    sm = w * vec(d2)' .+ ang2 # (np, ne)
    hderiv = sum(@. wm_deltaD * sin(sm); dims = 2) # (np, ne) -> (np,1)
    srm = @. mod2pi(sm + π) - π
    hcurv = sum(@. wm_deltaD2 * sinc(srm); dims = 2)
    return (vec(hderiv), vec(hcurv), sm)
end


# eqn (7) of Lin&Fessler
function decomp(w, relamp, echotime, zdata, γwf)
    (np,ne) = size(zdata)
    x = zeros(ComplexF32, 2, np) # water,fat
    for ip in 1:np
        B = @. γwf * cis(w[ip] * echotime) # (ne,2)
        x[:,ip] .= B \ zdata[ip,:] # (2,)
    end
    return x
end
