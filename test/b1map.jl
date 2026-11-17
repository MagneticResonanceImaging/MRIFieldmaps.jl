# test/b1map.jl

using MRIFieldmaps: roughness_penalty, b1_fit
using Test: @test, @testset, @test_throws, @inferred

@testset "b1map.jl" begin
    # We should expect no roughness since the matrix is full of the same values.
    z1 = ones(3, 3) * 5.0
    @test isapprox(roughness_penalty(z1), 0.0) 

    # Test with the middle pixel being different.
    z2 = zeros(3, 3)
    z2[2, 2] = 10.0
    expected = (-10)^2 + (-10)^2 + (2*10)^2 + (-10)^2 + (-10)^2;
    @test isapprox(roughness_penalty(z2), expected)

    # See if the regularizer function can sum up the roughness costs from z1 and z2.
    zdims = (3, 3, 1)
    zks = cat(reshape(z1, zdims), reshape(z2, zdims), dims=3)
    @test isapprox(regularizer(zks), expected)

    beta = 0.5
    MAX_ITTER = 100
    M = 2
    K = 1
    Chi = ones(M, K)
    Chi[1] = 1
    Chi[2] = 2 
    N1 = 10
    N2 = 20
    Y = ones(N1, N2, M)
    F = sin

    zks, fjs = b1_fit(beta, Y, Chi, F, MAX_ITTER)
    @test size(zks) == (N1, N2, K)
    @test size(fjs) == (N1, N2)
    @test eltype(zks) == ComplexF64
    @test eltype(fjs) == ComplexF64
end
