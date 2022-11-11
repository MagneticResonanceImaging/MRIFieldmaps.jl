# test/b0scale.jl

using MRIfieldmap: b0scale
using Test: @test, @testset, @inferred
#using Unitful: todo

@testset "b0scale" begin
    etime = [0, 2, 10] * 1f-3 # echo times
    ne = length(etime)
    dims = (16,15)

    # single coil
    ydata = randn(ComplexF32, dims..., ne)

    (yscaled, scalefactor) = b0scale(ydata, etime) 
    @test scalefactor isa Float32
    @test yscaled isa Array{ComplexF32}
    @test size(yscaled) == size(ydata)

    # multi coil
    nc = 2
#   smaps = 9 .+ randn(ComplexF32, dims..., nc, ne) # todo
    ydata = randn(ComplexF32, dims..., nc, ne)

    (yscaled, scalefactor) = b0scale(ydata, etime) 
    @test scalefactor isa Float32
    @test yscaled isa Array{ComplexF32}
    @test size(yscaled) == size(ydata)
end
