# test/b0scale.jl

using MRIFieldmaps: b0scale, div0
using Test: @test, @testset, @inferred
using Unitful: s

@testset "div0" begin
    @inferred div0(2s, 1f0s)
    @inferred div0(2//1, 1f0s)
end

@testset "b0scale" begin
    u = 1s
    echotime = [0, 2, 10] * 1f-3u # echo times in sec
    ne = length(echotime)
    dims = (16,15)

    # single coil
    ydata = randn(ComplexF32, dims..., ne)

    (yscaled, scalefactor) = b0scale(ydata, echotime)
    @test scalefactor isa Float32
    @test yscaled isa Array{ComplexF32}
    @test size(yscaled) == size(ydata)

    # multi coil
    nc = 2
#   smaps = 9 .+ randn(ComplexF32, dims..., nc, ne)
    ydata = randn(ComplexF32, dims..., nc, ne)

    (yscaled, scalefactor) = b0scale(ydata, echotime)
    @test scalefactor isa Float32
    @test yscaled isa Array{ComplexF32}
    @test size(yscaled) == size(ydata)
end
