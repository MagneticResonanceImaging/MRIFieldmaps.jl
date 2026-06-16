using MRIFieldmaps: MRIFieldmaps
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(MRIFieldmaps;
        deps_compat = (; ignore = [:LinearAlgebra, :SparseArrays]),
    )
end
