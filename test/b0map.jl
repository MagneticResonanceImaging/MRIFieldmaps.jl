# test/b0map.jl

using MRIfieldmap: b0map, b0scale
using Test: @test, @testset, @inferred
#using Unitful: todo
using ImageGeoms: ImageGeom, circle
using LinearAlgebra: norm
using Random: seed!

#=
using MIRTjim: jim
jim(:prompt, true)
jif = (args...; kwargs...) -> jim(args...; prompt=false, kwargs...)
=#

@testset "b0map" begin
#   @inferred b0map() # not type stable - too many "out" options

    etime = [0, 2, 10] * 1f-3 # echo times
    ig = ImageGeom( dims=(32,30) )
    mask = circle(ig)
    # simple test b0 fieldmap and image
    ftrue = -10 .+ 50 * exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))
    rmse = fhat -> round(norm((fhat - ftrue) .* mask) / sqrt(count(mask)); digits=1)
    flim = (-20, 40)
    xtrue = mask +
        (abs.(axes(ig)[1])/ig.dims[1] .+ abs.(axes(ig)[2]')/ig.dims[2] .< 0.3)

#=
function showit(fhat)
    jim(jif(ftrue, "ftrue"), jif(mask), jif(xtrue, "xtrue"),
        jif(finit.*mask, "finit"),
        jif(fhat, "fhat"; clim=flim, xlabel="$(rmse(fhat))"),
        jif(fhat-ftrue, "err"),
    )
end
=#

    # single coil 2D case

    ydata = [xtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in etime]
    ydata = cat(dims=3, ydata...)
    seed!(0)
    ydata .+= 0.03 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, etime)

    finit = angle.(ydata[:,:,2] .* conj(ydata[:,:,1])) / (etime[2] - etime[1]) / 2f0π
    finit .*= mask

    for track in (false, true)
        (fhat, _, _) = b0map(finit, ydata, etime; mask,
            niter=5, track, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    for precon in (:I, :diag, :chol, :ichol)
        (fhat, _, _) = b0map(finit, ydata, etime; mask,
            niter=5, track=false, precon, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    for gamma_type in (:FR, :PR)
        (fhat, _, _) = b0map(finit, ydata, etime; mask,
            niter=5, track=false, gamma_type, chat=false)
        @test maximum(abs, (fhat - ftrue) .* mask) < 2
    end

    # water-fat
    (fhat, _, out) = b0map(finit, ydata, etime; mask, niter=5, df=500)
#   @test maximum(abs, (fhat - ftrue) .* mask) < 5
    @test rmse(fhat) < 2
    @test fhat isa Matrix{Float32}
    @test out.xw isa Matrix{ComplexF32} # todo: test better
    @test out.xf isa Matrix{ComplexF32}


    # multiple coil 2D case

    smap = cat(fill(1f0+2im, ig.dims), fill(3-1im, ig.dims); dims=3) # (dims..., nc)

    ydata = [xtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in etime]
    ydata = cat(dims=3, ydata...) # (dims..., ne)
#   ydata = ydata .* reshape(smap, ig.dims..., 1, :)
    ydata = reshape(ydata, ig.dims..., 1, :) # (dims..., 1, ne)
    ydata = smap .* ydata
    seed!(0)
    ydata .+= 0.04 * randn(ComplexF32, size(ydata))
    ydata, _ = b0scale(ydata, etime)
    finit = angle.(ydata[:,:,1,2] .* conj(ydata[:,:,1,1])) / (etime[2] - etime[1]) / 2f0π

    (fhat, times, out) = b0map(finit, ydata, etime; mask, smap,
        niter=5, chat=false)
    @test maximum(abs, (fhat - ftrue) .* mask) < 2

    (fhat, times, out) = b0map(finit, ydata[:,:,1,:], etime; mask,
        # smap=ones(ComplexF32,ig),
        niter=9, chat=false)
    @test maximum(abs, (fhat - ftrue) .* mask) < 2
end
