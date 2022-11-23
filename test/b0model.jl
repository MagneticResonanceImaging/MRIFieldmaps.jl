# test/b0model.jl

using MRIFieldmaps: b0model
using Unitful: s
using Test: @test, @testset, @inferred


@testset "b0model" begin
    dims = (8,9)
    u = 1s
    echotime = [0, 2, 10] * 1f-3u # echo times in s
    ne = length(echotime)

    T = Float32
    Tc = complex(T)
    fmap = (10 .+ 20 * randn(T, dims)) / 1u
    nc = 2
    smap = randn(Tc, dims..., nc)
    xw = randn(Tc, dims)
    xf = randn(Tc, dims)
    relax = rand(T, dims) / 1u

    images = @inferred b0model(fmap, xw, echotime)
    @test size(images) == (dims..., 1, ne)
    @test images isa Array{Tc}

    images = @inferred b0model(fmap, xw, echotime; smap, relax)
    @test size(images) == (dims..., nc, ne)
    @test images isa Array{Tc}


    df = [1f0/u]
    relamp = [1f0]

    images = @inferred b0model(fmap, xw, echotime; df, relamp)
    @test size(images) == (dims..., 1, ne)
    @test images isa Array{Tc}

    images = @inferred b0model(fmap, xw, echotime; df, relamp, smap)
    @test size(images) == (dims..., nc, ne)
    @test images isa Array{Tc}

    images = @inferred b0model(fmap, xw, echotime; df, relamp, smap, relax)
    @test size(images) == (dims..., nc, ne)
    @test images isa Array{Tc}
end
