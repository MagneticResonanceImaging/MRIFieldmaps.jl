# test/combine.jl

using MRIFieldmaps: coil_combine
using Test: @test, @testset, @inferred

@testset "combine" begin
    nc, ne = 2, 3
    dims = (16,15)
    ydata = randn(ComplexF32, dims..., nc, ne)
    smap = 9 .+ randn(ComplexF32, dims..., nc)

    (zdata, sos) = @inferred coil_combine(ydata, smap)
    @test zdata isa AbstractArray{ComplexF32}
    @test sos isa AbstractArray{Float32}
    @test size(zdata) == (dims..., ne)
    @test size(sos) == dims
end
