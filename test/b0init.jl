# test/b0init.jl

using MRIfieldmap: b0init
using Test: @test, @testset, @inferred
#using Unitful: todo

@testset "b0init" begin
    echotime = [0, 2, 10] * 1f-3 # echo times
    ne = length(echotime)
    dims = (16,15)

    # single coil
    ydata = randn(ComplexF32, dims..., 1, ne)

    f1 = @inferred b0init(ydata, echotime) 
    @test f1 isa Array{Float32}
    @test size(f1) == dims

    # multi coil
    nc = 2
    ydata = randn(ComplexF32, dims..., nc, ne)
    smaps = 9 .+ randn(ComplexF32, dims..., nc)
    f2 = @inferred b0init(ydata, echotime) 
    @test f2 isa Array{Float32}
    @test size(f2) == dims
end
