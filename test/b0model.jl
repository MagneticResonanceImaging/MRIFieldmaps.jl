# test/b0model.jl

using MRIFieldmaps: b0model, fat_model
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
    xwater = randn(Tc, dims)
    xfat = randn(Tc, dims)
    relax = rand(T, dims) / 1u

    images = @inferred b0model(fmap, xwater, echotime)
    @test size(images) == (dims..., 1, ne)
    @test images isa Array{Tc}

    images = @inferred b0model(fmap, xwater, echotime; smap, relax)
    @test size(images) == (dims..., nc, ne)
    @test images isa Array{Tc}


    fat = fat_model(; sec=1f0u)
#   df = [1f0/u]
#   relamp = [1f0]

    images = @inferred b0model(fmap, xwater, echotime; fat..., xfat)
    @test size(images) == (dims..., 1, ne)
    @test images isa Array{Tc}

    images = @inferred b0model(fmap, xwater, echotime; fat..., xfat, smap)
    @test size(images) == (dims..., nc, ne)
    @test images isa Array{Tc}

    images = @inferred b0model(fmap, xwater, echotime; fat..., xfat, smap, relax)
    @test size(images) == (dims..., nc, ne)
    @test images isa Array{Tc}
end
