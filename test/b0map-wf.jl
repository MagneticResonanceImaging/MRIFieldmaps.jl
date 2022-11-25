# test/b0map-wf.jl (B0 fieldmap in water/fat separation case)

using MRIFieldmaps: b0map, b0model, b0init, b0scale
using Test: @test, @testset, @test_throws, @inferred
using Unitful: s
using ImageGeoms: ImageGeom, circle
using LinearAlgebra: norm
using Random: seed!


@testset "b0map-wf" begin
    seed!(0)
    u = 1s
    echotime = [0.0015 0.0038 0.0061 0.0084 0.0106 0.0129 0.0152 0.0174] * u
    echotime = Float32.(vec(echotime)) # from PKdata5.mat for 1.5 T

    nx, ny = (32, 30)
    ig = ImageGeom( dims=(nx,ny) )
    mask = circle(ig)

    # simple B0 fieldmap for testing
    ftrue = (-10 .+ 50 * exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))) / 1u
    rmse = fhat -> round(norm((fhat - ftrue) .* mask) / sqrt(count(mask)); digits=1)

    # water and fat images as ramps with random phases
    T = Float32
    xwtrue = mask .* (T.((1:nx) / nx) * ones(T, ny)') * 10 .* cis.(rand(T, nx,ny))
    xftrue = mask .* (ones(T, nx) * T.((1:ny)' / ny)) * 10 .* cis.(rand(T, nx,ny))

    # multi-species fat model parameters from notes.pdf in
    # https://www.ismrm.org/workshops/FatWater12/data.htm
    fieldstrength = 1.5 # Tesla
    gyro = 42.58/1u # gyromagnetic ratio in Hz/Tesla
    df = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60] *
        gyro * fieldstrength # Hz fat shift
    relamp = Float32[0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    smap = [fill(1f0+2im, ig.dims), fill(3-1im, ig.dims)] # nc=2 coils
    smap = cat(smap...; dims=3) # (dims..., nc)
    smap .*= mask

    ytrue = b0model(ftrue, xwtrue, echotime; smap, df, relamp, xf=xftrue)
    ydata = ytrue + 2 * randn(complex(T), size(ytrue))
    @show snr = 20*log(norm(ytrue) / norm(ydata - ytrue)) # 40 dB
    ydata, scale = b0scale(ydata, echotime) # todo: make internal to b0map!

    (fhat, _, out) = b0map(ydata, echotime;
        # b0init_args = (; fband=80, threshold=0.01),
        order = 2, l2b = -3,
        mask, smap, niter=8, df, relamp, track=true, precon=:diag)

    @test maximum(abs, (fhat - ftrue) .* mask) < 2/u
    @test fhat isa Matrix{eltype(oneunit(T) / 1u)}
    @test out.xw isa Matrix{complex(T)}
    @test out.xf isa Matrix{complex(T)}
    @test norm(scale*out.xw - xwtrue) / norm(xwtrue) < 0.06
    @test norm(scale*out.xf - xftrue) / norm(xftrue) < 0.06
end
