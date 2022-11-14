# test/b0map.jl

using MRIfieldmap: b0map, b0scale, b0init
using Test: @test, @testset, @test_throws, @inferred
using Unitful: s
using ImageGeoms: ImageGeom, circle
using LinearAlgebra: norm
using Random: seed!

#=
using MIRTjim: jim
jim(:prompt, true)
jif = (args...; kwargs...) -> jim(args...; prompt=false, kwargs...)
=#

@testset "b0map" begin
#   @inferred b0map() # not type stable - too many "out" options

#   u = 1s
    u = 1 # units not supported by lldl method
    echotime = [0, 2, 10] * 1f-3u # echo times
    ne = length(echotime)
    ig = ImageGeom( dims=(32,30) )
    mask = circle(ig)
    # simple test b0 fieldmap and image
    ftrue = exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))
    ftrue = (-10 .+ 50 * ftrue) / 1u
    rmse = fhat -> round(norm((fhat - ftrue) .* mask) / sqrt(count(mask)); digits=1)
    flim = (-20/u, 40/u)
    xtrue = mask +
        (abs.(axes(ig)[1])/ig.dims[1] .+ abs.(axes(ig)[2]')/ig.dims[2] .< 0.3)

#=
function showit(fhat)
    jim(jif(ftrue, "ftrue"), jif(mask), jif(xtrue, "xtrue"),
        jif(finit.*mask, "finit"),
        jif(fhat, "fhat"; clim=flim, xlabel="$(rmse(fhat))"),
        jif(fhat-ftrue, "err"),
    )
end
=#

    # single coil 2D case

    ydata = [xtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in echotime]
    ydata = cat(dims=3, ydata...)
    seed!(0)
    ydata .+= 0.03 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, echotime)

    finit = b0init(reshape(ydata, ig.dims..., 1, ne), echotime)
    finit .*= mask

    for track in (false, true)
        (fhat, _, _) = b0map(finit, ydata, echotime; mask,
            niter=5, track, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    for precon in (:I, :diag, :chol, :ichol)
        (fhat, _, _) = b0map(finit, ydata, echotime; mask,
            niter=5, track=false, precon, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    for gamma_type in (:FR, :PR)
        (fhat, _, _) = b0map(finit, ydata, echotime; mask,
            niter=5, track=false, gamma_type, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    # multiple coil 2D case

    smap = cat(fill(1f0+2im, ig.dims), fill(3-1im, ig.dims); dims=3) # (dims..., nc)

    ydata = [xtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in echotime]
    ydata = cat(dims=3, ydata...) # (dims..., ne)
#   ydata = ydata .* reshape(smap, ig.dims..., 1, :)
    ydata = reshape(ydata, ig.dims..., 1, :) # (dims..., 1, ne)
    ydata = smap .* ydata
    seed!(0)
    ydata .+= 0.04 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, echotime)
#   finit = angle.(ydata[:,:,1,2] .* conj(ydata[:,:,1,1])) / (echotime[2] - echotime[1]) / 2f0π
    finit = b0init(ydata, echotime; smap)

    let
        (fhat, times, out) = b0map(finit, ydata, echotime; mask, smap,
            niter=5, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    let
        (fhat, times, out) = b0map(finit, ydata[:,:,1,:], echotime; mask,
            # smap=ones(ComplexF32,ig),
            niter=9, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end
end
