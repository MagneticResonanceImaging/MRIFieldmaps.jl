# test/spdiff2.jl

using MRIFieldmaps: spdiff2, spdiff
using LinearAlgebra: I
using Test: @test, @testset, @test_throws, @inferred

@testset "spdiff2" begin
    @inferred spdiff2(3)
    @inferred spdiff((3,4,5); order=2)

    @test spdiff2(3; ending=:zero) == [2 -1 0; -1  2 -1; 0 -1 2]
    @test spdiff2(3; ending=:first) == [1 -1 0; -1  2 -1; 0 -1 1]
    @test spdiff2(4; ending=:remove) == [0 0 0 0; -1  2 -1 0; 0 -1 2 -1; 0 0 0 0]
    @test_throws String spdiff2(3; ending=:bad)

    @test spdiff((4,5,6); order=2)[1] == kron(I(6*5), spdiff2(4))
    @test spdiff((4,5,6); order=2)[2] == kron(I(6), spdiff2(5), I(4))
    @test spdiff((4,5,6); order=2)[3] == kron(spdiff2(6), I(4*5))

    @test spdiff((4,5); order=2)[1] == kron(I(5), spdiff2(4))
    @test spdiff((4,5); order=2)[2] == kron(spdiff2(5), I(4))
end
