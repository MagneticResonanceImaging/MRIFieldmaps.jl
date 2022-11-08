# test/runtests.jl

using Test: @test, @testset, detect_ambiguities
import MRIfieldmap

@testset "MRIfieldmap" begin
    @test isempty(detect_ambiguities(MRIfieldmap))
end

include("spdiff1.jl")
include("spdiff2.jl")
