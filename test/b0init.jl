# test/b0init.jl

using MRIfieldmap: b0init
using Test: @test, @testset, @inferred
using Unitful: s

@testset "b0init" begin
    u = 1s
    echotime = [0, 2, 10] * 1f-3u # echo times in s
    ne = length(echotime)
    dims = (16,15)

    # single coil
    ydata = randn(ComplexF32, dims..., 1, ne)

    f1 = @inferred b0init(ydata, echotime) 
    @test size(f1) == dims
    @test f1*u isa Array{Float32}

    # multi coil
    nc = 2
    ydata = randn(ComplexF32, dims..., nc, ne)
    smaps = 9 .+ randn(ComplexF32, dims..., nc)
    f2 = @inferred b0init(ydata, echotime) 
    @test size(f2) == dims
    @test f2*u isa Array{Float32}
end
