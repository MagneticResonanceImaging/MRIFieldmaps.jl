# test/runtests.jl

using Test: @test, @testset, detect_ambiguities
import MRIfieldmaps

@testset "MRIfieldmaps" begin
    @test isempty(detect_ambiguities(MRIfieldmaps))
end

include("spdiff1.jl")
include("spdiff2.jl")
include("b0init.jl")
include("b0scale.jl")
include("combine.jl")
include("b0map.jl")
include("b0map-wf.jl")
