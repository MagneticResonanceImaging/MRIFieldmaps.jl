# b0map.jl

using LinearAlgebra: norm, cholesky
using ImageGeoms: embed
using SparseArrays: spdiagm, diag
using LimitedLDLFactorizations: lldl
using MRIfieldmap: spdiff


"""
    (fhat, times, out) = b0map(finit, ydata, delta; kwargs...)

Field map estimation from multiple (`ne ≥ 2`) echo-time images,
using preconditioned nonlinear CG (NCG) with a monotonic line search.
This code works with images of arbitrary dimensions (2D, 3D, etc.),
and with multiple coils.

The cost function for the single-coil case is:
`cost(w) = ∑_{j=1}^{#voxel} ∑_{m=1}^ne ∑_{n=1}^ne
    |y_{mj} y_{nj}| wj (1 - cos(w_j * (t_m - t_n) + ∠y_{ni} - ∠y_{mj})) + R(w)`
where `t_n` denotes the echo time of the `n`th scan and
`R(w) = 0.5 * | C * w |^2`
is a quadratic roughness regularizer
based on 1st-order or 2nd-order finite differences.

The initial field map `finit` and output `fhat` are field maps in Hz,
but internally the code works with `ω = 2π f` (rad/s).

# In
- `finit (dims)` initial fieldmap estimate in Hz
- `ydata (dims..., [nc,] ne)` `ne` sets of complex images for `nc` coils
- `delta (ne)` vector of `ne` echo time offsets

# Options
- `smap (dims...[, nc])` complex coil maps, default `ones(size(finit))`
- `mask (dims...)` logical reconstruction mask; default: `trues(size(finit))`
- `ninner` # of inner iterations for monotonic line search
  default: `3` inner iterations
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
- `df` Δf value in water-fat imaging (def: 0) units Hz, e.g., 440 at 3T
- `relamp` relative amplitude in multipeak water-fat (def: `1`)
- `lldl_args::NamedTuple` options for `lldl`, default: `(;memory=2)`
- `track::Bool` if `true` then track cost and save all iterations (def: `false`)
- `chat::Bool = true` # `@info` updates each iteration
- `chat_iter::Int = 10` # print progress report every few iterations.

# Out
- `fhat` final fieldmap estimate in Hz
- `times (niter+1)` wall time for each iteration
- `out::NamedTuple` that contains:
   * `(xw, xf) (dims)` water / fat images if `!iszero(df)`
- if `track == true` then `out` also contains:
   * `costs (niter+1)` (nonconvex) cost for each iteration
   * `fhats (dims, niter+1)` fieldmap estimates every iteration

The algorithm is based on the paper:
C Y Lin, J A Fessler,
"Efficient Regularized Field Map Estimation in 3D MRI", IEEE TCI 2020
http://doi.org/10.1109/TCI.2020.3031082
http://arxiv.org/abs/2005.08661
"""
function b0map(
    finit::AbstractArray{<:Real},
    ydata::Array{<:Complex},
    delta::AbstractVector{<:Real},
    ;
    smap::AbstractArray{<:Complex} = ones(ComplexF32, size(finit)..., 1),
    mask::AbstractArray{<:Bool} = trues(size(finit)),
    kwargs...
)

    Base.require_one_based_indexing(delta, finit, mask, smap)

    dims = size(finit)
    ndim = length(dims)
    if ndims(smap) == ndim
        smap = reshape(smap, size(smap)..., 1) # nc=1 special case
    end
    nc = size(smap, ndim+1) # ≥ 1

    # handle ydata size (dims..., ne) when nc=1
    ndims(ydata) ∈ (ndim+1, ndim+2) ||
        throw("ydata ndims=$(ndims(ydata)) vs ndim=$ndim")
    size(ydata)[1:ndim] == dims || throw("bad ydata size")
    if nc == 1 && ndims(ydata) == ndim + 1
        ydata = reshape(ydata, dims..., 1, size(ydata, ndim+1))
    end
    ne = size(ydata)[ndim+2]

    # check dimensions
    ne == length(delta) || throw("need delta to have length ne=$ne")
    dims == size(mask) || throw("bad mask size $(size(mask)) vs dims=$dims")
    (dims..., nc) == size(smap) ||
        throw("bad smap size $(size(smap)) vs dims=$dims & nc=$nc")

    (fhat, times, out) = b0map(
        finit[mask],
        reshape(ydata, :, nc, ne)[vec(mask), :, :],
        delta,
        reshape(smap, :, nc)[vec(mask), :],
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
- `ydata (np, nc, ne)` `ne` sets of measurements for `nc` coils
- `delta (ne)` vector of `ne` echo time offsets
- `smap (np, nc)` coil maps
- `mask (N)` logical reconstruction mask
"""
function b0map(
    finit::AbstractVector{<:Real},
    ydata::AbstractArray{<:Complex},
    delta::AbstractVector{<:Real},
    smap::AbstractArray{<:Complex},
    mask::AbstractArray{<:Bool},
    ;
    ninner::Int = 3, # subiterations
    niter::Int = 30,
    order::Int = 2,
    l2b::Real = -6,
    gamma_type::Symbol = :PR,
    precon::Symbol = :ichol,
    reset::Real = Inf,
    df::RealU = 0,
    relamp::Real = 1,
    chat::Bool = true,
    chat_iter::Int = 10, # progress report this often
    track::Bool = false,
    lldl_args::NamedTuple = (; memory = 2),
)

    t0 = time() # start timer

    # check dimensions
    (np, nc, ne) = size(ydata)
    ne == length(delta) || throw("need delta to have length ne=$ne")
    np == length(finit) || throw("need finit to have length np=$np")
    (np, nc) == size(smap) || throw("need smap to have size (np,nc)=($np,$nc)")
    count(mask) == np || throw("bad mask count")

    # sparse finite-difference regularization matrix (?,np)
    C = vcat(spdiff(size(mask); order)...)

    # remove all rows of C that involve non-mask pixels to avoid "leaking"
    good = iszero.(abs.(C) * .!vec(mask))
    C = C[good,:]
    C = C[:,vec(mask)] # (?,np) apply mask
    C = sqrt(2f0^l2b) * C
    CC = C' * C

    # calculate the magnitude and angles used in the data-fit curvatures
    sjtotal = sum(abs2, smap; dims=2) # (np,1)
    angy = angle.(ydata)
    angs = angle.(smap)
    if !iszero(df)
        Gamma = phiInv(relamp, df, delta) # (L,L)
    end

    nset = cumsum(1:ne-1)[end]

    wj_mag = zeros(Float32, np, nset, nc, nc)
    d2 = zeros(Float32, 1, nset, 1, 1) # trick
    ang2 = zeros(Float32, np, nset, nc, nc)

    set = 0

    # Precompute data-dependent magnitude and phase factors.
    for j in 1:ne # for each pair of scans: "m,n" in (3) in the paper
        for i in 1:ne
            (i ≥ j) && continue # only need one pair of differences
            set += 1
            d2[set] = delta[i] - delta[j]
            for c in 1:nc
                for d in 1:nc
                    wj_mag[:,set,c,d] .= abs.(smap[:,c] .* conj(smap[:,d]) .*
                        conj(ydata[:,c,i]) .* ydata[:,d,j])
                    # difference of the echo times and angles
                    ang2[:,set,c,d] .= angs[:,c] - angs[:,d] +
                                       angy[:,d,j] - angy[:,c,i]
                    if !iszero(df)
                        wj_mag[:,set,c,d] .*= abs.(Gamma[i,j])
                        ang2[:,set,c,d] .+= angle(Gamma[i,j])
                    end
                end
            end
        end
    end

    # compute |s_c s_d' y_dj' y_ci| /L/s * (tj - ti)^2
    sjtotal[sjtotal .== 0] .= 1 # avoid outside mask = 0
    wj_mag ./= sjtotal # todo: use div0 ?
    if iszero(df)
        wj_mag ./= ne # eqn. (4) in paper for non-fat case
    end
    wm_deltaD = wj_mag .* d2
    wm_deltaD2 = wj_mag .* abs2.(d2)
    any(isnan, ang2) && warn("isnan in ang2")
    ang2[isnan.(ang2)] .= 0 # avoid atan causing nan

    # prepare output variables
    times = zeros(niter+1)
    if track
        out_fs = zeros(length(finit), niter+1)
        out_fs[:,1] .= finit
        costs = zeros(niter+1)
        # todo store initial cost
#=
        sm = w * vec(d2)' .+ ang2 # (np, ne, nc, nc)
        cost0d = sum(wj_mag .* (1 .- cos.(sm)))
        cost0r = 0.5 * norm(C*w)^2
        cost0 = cost0d + cost0r
=#
    end

    if precon === :diag
        dCC = diag(CC)
        dCC = Vector{Float32}(dCC)
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
# todo: do this after update
# todo: validate derivatives of this fancy cost
            costs[iter] = sum(wj_mag .* (1 .- cos.(sm))) + 0.5 * norm(C*w)^2
            chat && @info "iter $(iter-1), cost $(costs[iter])"
        end

        # apply preconditioner

        if precon === :I
            npregrad = ngrad

        elseif precon === :diag
            H = hcurv + dCC
            npregrad = ngrad ./ H

        elseif precon === :chol
            H = spdiagm(hcurv) + CC
            PC = cholesky(H) # may run out of memory for large problems
            npregrad = Vector{Float32}(PC \ ngrad)
            # preceding cast is due to CHOLMOD limitation as of 2022-11
            # https://github.com/JuliaSparse/SparseArrays.jl/issues/111

        elseif precon === :ichol
            H = spdiagm(hcurv) + CC
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

        # check if correct descent direction
        if ddir' * grad > 0
            if !warned_dir
                warned_dir = true
                @warn "wrong direction so resetting"
                @warn "<ddir,grad>=$(ddir'*grad), |ddir|=$(norm(ddir)), |grad|=$(norm(grad))"
            end
            # reset direction if not descending
            ddir = npregrad
            oldinprod = 0
        end

        # step size in search direction
        Cdir = C * ddir # caution: can be a big array for 3D problems

        # compute the monotonic line search using quadratic surrogates
        CdCd = Cdir' * Cdir
        CdCw = ddir' * CCw
        step = 0
        for is in 1:ninner

            # compute the curvature and derivative for subsequent steps
            if step != 0
                (hderiv,hcurv) = Adercurv(d2, ang2, wm_deltaD, wm_deltaD2, w + step * ddir)
            end

            # compute numer and denom of the Huber's algorithm based line search
            denom = abs2.(ddir)' * hcurv + CdCd
            numer = ddir' * hderiv + (CdCw + step * CdCd);

            if denom == 0
                @warn "found exact solution??? step=0 now!?"
                step = 0
            else
                # update line search
                step = step - numer / denom
            end

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
        costs[niter+1] = sum(wj_mag .* (1 .- cos.(sm))) + 0.5 * norm(C*w)^2
    #   chat && @info ' ite: %d , cost: %f3\n', iter, cost(iter+1))
        out = (fhats = out_fs, costs)
    else
        out = (; )
    end

    # output water & fat images
    if !iszero(df)
        x = decomp(w, relamp, df, delta, smap, ydata)
        out = merge(out, (xw = x[1,:], xf = x[2,:]))
    end

    return w / 2f0π, times, out # return Hz
end


# compute the data-fit derivatives and curvatures as in Funai 2008 paper
function Adercurv(d2, ang2, wm_deltaD, wm_deltaD2, w)
    sm = w * vec(d2)' .+ ang2 # (np, ne, nc, nc)
    tmp = wm_deltaD .* sin.(sm) # (np, ne, nc, nc)
    hderiv = 2 * sum(wm_deltaD .* sin.(sm); dims = 2:4)
    srm = @. mod2pi(sm + π) - π
    hcurv = 2 * sum(wm_deltaD2 .* sinc.(srm); dims = 2:4)
    return (vec(hderiv), vec(hcurv), sm)
end


function phiInv(relamp, df, delta)
    ne = length(delta)
    phi = [ones(ne) sum(relamp .* cis.(2π*delta*df); dims=2)] # (n,2)
    Gamma = phi * inv(phi'*phi) * phi'
    return Gamma
end


function decomp(w, relamp, df, delta, smap, ydata)
    (np,nc,ne) = size(ydata)
    phi = [ones(ne) sum(relamp .* cis.(2π*delta*df); dims=2)] # (ne,2)
    x = zeros(ComplexF32, 2,np) # water,fat
    for ip in 1:np
        B = phi .* cis.(w[ip] * delta) # (ne,2)
        B = kron(smap[ip,:], B) # (ne*nc,2) with ne varying fastest
        yc = transpose(ydata[ip,:,:]) # (nc, ne) -> (ne, nc)
        x[:,ip] .= B \ vec(yc) # (2,)
    end
    return x
end
