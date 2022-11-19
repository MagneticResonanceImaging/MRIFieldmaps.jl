# test/b0init.jl

using MRIfieldmap: b0init, b0model
using Random: seed!
using Unitful: s
using Test: @test, @testset, @inferred

@testset "b0init" begin
    seed!(0)
    u = 1s
    echotime = [0, 2, 10] * 1f-3u # echo times in s
    ne = length(echotime)
    dims = (10,8)

    # single coil

    ftrue = 20 * randn(Float32, dims...) / 1u
    xtrue = 2 .+ rand(ComplexF32, dims)
    ytrue = b0model(ftrue, xtrue, echotime)
    ydata = ytrue + 0.01f0 * randn(ComplexF32, size(ytrue))

    f0 = @inferred b0init(ytrue, echotime)
    @test f0 ≈ ftrue # noiseless case

    f1 = @inferred b0init(ydata, echotime) # noisy
    @test size(f1) == dims
    @test f1*u isa Array{Float32}
    @test maximum(abs, f1 - ftrue) < 2/u


    # multi coil

    nc = 2
    smap = 9 .+ randn(ComplexF32, dims..., nc)
    ytrue = b0model(ftrue, xtrue, echotime; smap)
    ydata = ytrue + 0.01f0 * randn(ComplexF32, size(ytrue))

    f2 = @inferred b0init(ytrue, echotime; smap)
    @test f2 ≈ ftrue

    f3 = @inferred b0init(ydata, echotime; smap)
    @test maximum(abs, f3 - ftrue) < 1/u
    @test f3*u isa Array{Float32}

    # discrete search approach

    f4 = @inferred b0init(ytrue, echotime; smap, df=[0f0/u], nf=201)
    @test maximum(abs, f4 - ftrue) < 2/u # due to quantization error
    f5 = @inferred b0init(ydata, echotime; smap, df=[0f0/u]) # noisy
    @test maximum(abs, f5 - ftrue) < 3/u
end
