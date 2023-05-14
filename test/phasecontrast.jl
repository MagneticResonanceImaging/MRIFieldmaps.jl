# test/phasecontrast.jl

using MRIFieldmaps: b0map, phasecontrast
using Random: seed!
using Test: @test, @testset, @test_throws, @inferred
using Unitful: s, ms, Hz

@testset "phasecontrast" begin
    seed!(0)
    function sensitivity_map(nx, ny, cx, cy, σ)
        s = [exp(-((x - cx)^2 + (y - cy)^2) / σ^2) for x in 1:nx, y in 1:ny]
        return complex.(s ./ maximum(s))
    end

    # Specify the parameters for the sensitivity maps and then generate the maps.
    nx = ny = 64
    σ = 60
    centers = [
        (nx ÷ 2, -100),
        (nx ÷ 2, 100 + ny + 1),
        (-100, ny ÷ 2),
        (100 + nx + 1, ny ÷ 2),
    ]
    smap = cat([sensitivity_map(nx, ny, cx, cy, σ) for (cx, cy) in centers]...; dims = 3)

    # Create a disk-shaped object with uniform intensity (M0) throughout.
    radius = 25
    obj = [(x - nx÷2)^2 + (y - ny÷2)^2 < radius^2 for x in 1:nx, y in 1:ny]
    mask = obj .> 0 # Mask indicating object support

    # Create a linearly (spatially) varying field map.
    ftrue = repeat(range(-50, 50, nx), 1, ny) * Hz
    ftrue .*= mask # Mask out background voxels

    # Specify parameters for the simulated multi-echo, multi-coil data.
    TE = (4.0ms, 6.3ms)
    T2 = 40ms
    σ_noise = 0.005; # Noise standard deviation

    # Make a function for creating the simulated data
    # for a given echo time.
    function make_data(TE)
        data = @. smap * obj * exp(-TE / T2) * cispi(2TE * ftrue) # Noiseless
        @. data += σ_noise * randn.(ComplexF64) # Add complex Gaussian noise
        return data
    end

    ydata = cat(make_data.(TE)...; dims = 4) # Create the simulated data.
    l2b = -26 # Regularization parameter is β = 2^l2b
    precon = :diag

    # Use phase contrast-based approach
    fpc_iterative = b0map(ydata, TE; smap = nothing, l2b, precon)[1] .* mask

    # Do conventional (non-iterative) phase contrast B0 mapping
    fpc = phasecontrast(ydata, TE) .* mask

    @test maximum(abs, fpc - ftrue) < 5Hz
    @test maximum(abs, fpc_iterative - ftrue) < 5Hz
end
