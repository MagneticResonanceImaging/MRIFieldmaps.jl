# test/b0map-wf.jl (B0 fieldmap in water/fat separation case)

using MRIfieldmap: b0map, b0init, b0scale
using Test: @test, @testset, @test_throws, @inferred
#using Unitful: s # lldl does not support units
using ImageGeoms: ImageGeom, circle
using LinearAlgebra: norm
using Random: seed!

#=
# debug:
using MIRTjim: jim
jim(:prompt, true)
jif = (args...; kwargs...) -> jim(args...; prompt=false, kwargs...)

function showit(ftrue, xwtrue, xftrue, finit, fhat, xw, xf;
 clim = (-20, 40),
 elim = (-20, 20),
)
    jim(
        jif(mask, "mask"),
        jif(ftrue, "ftrue"),
        jif(xwtrue, "xwtrue"),
        jif(xftrue, "xftrue"),

        jif(finit.*mask, "finit"; clim),
        jif(fhat, "fhat"; clim, xlabel="$(rmse(fhat))"),
        jif(xw, "xw"),
        jif(xf, "xf"),

        jif(finit-ftrue, "f init err"; clim=elim),
        jif(fhat-ftrue, "f err"; clim=elim),
        jif(xw-xwtrue, "xw err"),
        jif(xf-xftrue, "xf err"),
    )
end
=#


@testset "b0map-wf" begin
    seed!(0)
    echotime = [0, 2, 10] * 1f-3 # echo times
    nx, ny = (32, 30)
    ig = ImageGeom( dims=(nx,ny) )
    mask = circle(ig)

    # simple B0 fieldmap for testing
    ftrue = -10 .+ 50 * exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))
    rmse = fhat -> round(norm((fhat - ftrue) .* mask) / sqrt(count(mask)); digits=1)

    # water and fat images as ramps with random phases
    T = Float32
    xwtrue = mask .* (T.((1:nx) / nx) * ones(T, ny)') * 10 .* cis.(rand(T, nx,ny))
    xftrue = 0mask .* (ones(T, nx) * T.((1:ny)' / ny)) * 10 .* cis.(rand(T, nx,ny))

    df = [440f0] # Hz fat shift
    relamp = [1f0]
    ywtrue = [xwtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in echotime]
    ywtrue = cat(dims=3, ywtrue...)
    yftrue = [xftrue .* cis.(2f0π * (ftrue .+ df) * ΔTE) for ΔTE in echotime]
    yftrue = cat(dims=3, yftrue...)
    ydata = ywtrue + yftrue

    smap = [fill(1f0+2im, ig.dims), fill(3-1im, ig.dims)] # nc=2 coils
    smap = cat(smap...; dims=3) # (dims..., nc)
#   smap = ones(ComplexF32, nx, ny, 2)

    ydata = reshape(ydata, ig.dims..., 1, :) # (dims..., 1, ne)
    ydata = smap .* ydata

    ydata .+= 0.01 * randn(ComplexF32, size(ydata))
    ydata, scale = b0scale(ydata, echotime) # todo: make internal to b0map!

    # todo: need better init for water/fat
    finit = b0init(ydata, echotime; smap)
    finit .*= mask
#finit = ftrue .* mask # todo: debug

    (fhat, _, out) = b0map(finit, ydata, echotime;
        order = 1, l2b = -4, # todo
        mask, smap, niter=18, df, track=true, precon=:diag)

#   @test maximum(abs, (fhat - ftrue) .* mask) < 5
#   @test_throws ErrorException rmse(fhat) < 2 # todo
    @test fhat isa Matrix{Float32}
    @test out.xw isa Matrix{ComplexF32} # todo: test better
    @test out.xf isa Matrix{ComplexF32}
#   showit(ftrue, xwtrue, xftrue, finit, fhat, scale*out.xw, scale*out.xf)
end
