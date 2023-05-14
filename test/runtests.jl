# test/runtests.jl

using Test: @test, @testset, detect_ambiguities
import MRIFieldmaps

@testset "MRIFieldmaps" begin
    @test isempty(detect_ambiguities(MRIFieldmaps))
end

include("spdiff1.jl")
include("spdiff2.jl")
include("b0init.jl")
include("b0scale.jl")
include("combine.jl")
include("b0map.jl")
include("b0map-wf.jl")
include("phasecontrast.jl")
