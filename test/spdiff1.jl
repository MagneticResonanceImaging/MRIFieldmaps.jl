# test/spdiff1.jl

using MRIfieldmap: spdiff1, spdiff
using LinearAlgebra: I
using Test: @test, @testset, @inferred

@testset "spdiff1" begin
    @inferred spdiff1(3)
    @inferred spdiff((3,4,5); order=1)

    @test spdiff1(3; ending=:zero) == [1 0 0; -1 1 0; 0 -1 1]
    @test spdiff1(3; ending=:remove) == [0 0 0; -1 1 0; 0 -1 1]

    @test spdiff((4,5,6))[1] == kron(I(6*5), spdiff1(4))
    @test spdiff((4,5,6))[2] == kron(I(6), spdiff1(5), I(4))
    @test spdiff((4,5,6))[3] == kron(spdiff1(6), I(4*5))
end
