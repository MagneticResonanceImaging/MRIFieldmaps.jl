# test/b0map.jl

using MRIFieldmaps: b0map, b0scale, b0model, _check_descent!
using Test: @test, @testset, @test_throws, @inferred
using Unitful: s
using ImageGeoms: ImageGeom, circle
using LinearAlgebra: norm
using Random: seed!


@testset "b0map" begin

    _check_descent!([1], [1], 0, 0, false) # code coverage

#   @inferred b0map() # not type stable - too many "out" options

    u = 1s # test with units
    echotime = (0, 2, 10) .* 1f-3u # echo times
    ne = length(echotime)
    dims = (32,30)
    ig = ImageGeom( ; dims )
    mask = circle(ig)
    # simple test b0 fieldmap and image
    ftrue = exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))
    ftrue = (-10 .+ 50 * ftrue) / 1u
    rmse = fhat -> round(norm((fhat - ftrue) .* mask) / sqrt(count(mask)); digits=1)
    flim = (-20/u, 40/u)
    xtrue = mask +
        (abs.(axes(ig)[1])/ig.dims[1] .+ abs.(axes(ig)[2]')/ig.dims[2] .< 0.3)

    # single coil 2D case
    ydata = @inferred b0model(ftrue, xtrue, echotime)

    seed!(0)
    ydata .+= 0.03 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, echotime)

    for track in (false, true)
        (fhat, _, _) = b0map(ydata, echotime; mask,
            niter=5, track, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2/u
    end

    for precon in (:I, :diag, :chol, :ichol)
        (fhat, _, _) = b0map(ydata, echotime; mask,
            niter=5, track=false, precon, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2/u
    end

    @test_throws String b0map(ydata, echotime; mask, precon=:bad)

    for gamma_type in (:FR, :PR)
        (fhat, _, _) = b0map(ydata, echotime; mask,
            niter=5, track=false, gamma_type, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2/u
    end

    @test_throws String b0map(ydata, echotime; mask, gamma_type=:bad)

    # multiple coil 2D case

    smap = cat(fill(1f0+2im, ig.dims), fill(3-1im, ig.dims); dims=3) # (dims..., nc)
    ydata = @inferred b0model(ftrue, xtrue, echotime; smap)
    seed!(0)
    ydata .+= 0.04 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, echotime)
# todo: scale in b0map top level

    let
        (fhat, times, out) = b0map(ydata, echotime; mask, smap,
            niter=5, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2/u
    end

    let
        (fhat, times, out) = b0map(ydata, echotime; mask,
            niter=9, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2/u
    end
end
