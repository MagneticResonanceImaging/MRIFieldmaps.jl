# test/b0map-wf.jl (B0 fieldmap in water/fat separation case)

using MRIfieldmap: b0map, b0scale
using Test: @test, @testset, @test_throws, @inferred
#using Unitful: todo
using ImageGeoms: ImageGeom, circle
using LinearAlgebra: norm
using Random: seed!

#=
=#
# debug:
using MIRTjim: jim
jim(:prompt, true)
jif = (args...; kwargs...) -> jim(args...; prompt=false, kwargs...)

function showit(ftrue, xwtrue, xftrue, finit, fhat, xw, xf; clim = (-20, 40))
    jim(
        jif(mask, "mask"),
        jif(ftrue, "ftrue"),
        jif(xwtrue, "xwtrue"),
        jif(xftrue, "xftrue"),

        jif(finit.*mask, "finit"),
        jif(fhat, "fhat"; clim, xlabel="$(rmse(fhat))"),
        jif(xw, "xw"),
        jif(xf, "xf"),

        jif(finit-ftrue, "f init err"),
        jif(fhat-ftrue, "f err"),
        jif(xw-xwtrue, "xw err"),
        jif(xf-xftrue, "xf err"),
    )
end


@testset "b0map-wf" begin
end

    echotime = [0, 2, 10] * 1f-3 # echo times
    nx, ny = (32, 30)
    ig = ImageGeom( dims=(nx,ny) )
    mask = circle(ig)

    # simple test B0 fieldmap
    ftrue = -10 .+ 50 * exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))
    rmse = fhat -> round(norm((fhat - ftrue) .* mask) / sqrt(count(mask)); digits=1)

    # water and fat images as ramps with random phases
    T = Float32
    xwtrue = mask .* (T.((0:nx-1) / nx) * ones(T, ny)') * 10 .* cis.(rand(T, nx,ny))
    xftrue = mask .* (ones(T, nx) * T.((0:ny-1)' / ny)) * 10 .* cis.(rand(T, nx,ny))

    df = 440 # Hz fat shift
    ywtrue = [xwtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in echotime]
    ywtrue = cat(dims=3, ywtrue...)
    yftrue = [xftrue .* cis.(2f0π * (ftrue .+ df) * ΔTE) for ΔTE in echotime]
    yftrue = cat(dims=3, yftrue...)
    ydata = ywtrue .* yftrue

    smap = [fill(1f0+2im, ig.dims), fill(3-1im, ig.dims)] # nc=2 coils
    smap = cat(smap...; dims=3) # (dims..., nc)

    ydata = reshape(ydata, ig.dims..., 1, :) # (dims..., 1, ne)
    ydata = smap .* ydata

    seed!(0)
    ydata .+= 0.04 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, echotime)

    # todo: need better init for water/fat
    finit = angle.(ydata[:,:,1,2] .* conj(ydata[:,:,1,1])) / (echotime[2] - echotime[1]) / 2f0π
    finit .*= mask
finit = ftrue .* mask # todo: debug
    showit(ftrue, xwtrue, xftrue, finit, 0*ftrue, 0*xwtrue, 0*xftrue)

    (fhat, _, out) = b0map(finit, ydata, echotime; mask, smap, niter=15, df=440)
#   @test maximum(abs, (fhat - ftrue) .* mask) < 5
#   @test_throws ErrorException rmse(fhat) < 2 # todo
        @test fhat isa Matrix{Float32}
        @test out.xw isa Matrix{ComplexF32} # todo: test better
        @test out.xf isa Matrix{ComplexF32}
#showit(fhat)
    showit(ftrue, xwtrue, xftrue, finit, fhat, out.xw, out.xf)
#=
=#

#end
