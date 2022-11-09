# test/b0map.jl

using MRIfieldmap: b0map
using Test: @test, @testset, @inferred
#using Unitful: todo
using MIRTjim: jim
using ImageGeoms: ImageGeom, circle
using Random: seed!

@testset "b0map" begin
#   @inferred b0map() # not type stable - too many "out" options

    etime = [0, 2, 10] * 1f-3 # echo times
    ig = ImageGeom( dims=(32,30) )
    mask = circle(ig)
    # simple test b0 fieldmap and image
    ftrue = -10 .+ 50 * exp.(-0.1f0 * (abs2.(axes(ig)[1]) .+ abs2.(axes(ig)[2]')))
    xtrue = mask +
        (abs.(axes(ig)[1])/ig.dims[1] .+ abs.(axes(ig)[2]')/ig.dims[2] .< 0.3)

    # single coil case

    ydata = [xtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in etime]
    ydata = cat(dims=3, ydata...)
    seed!(0)
    ydata .+= 0.02 * randn(ComplexF32, size(ydata))

    finit = angle.(ydata[:,:,2] .* conj(ydata[:,:,1])) / (etime[2] - etime[1]) / 2f0π

    for track in (false, true)
        (fhat, _, _) = b0map(finit, ydata, etime; niter=5, l2b=-3, track, chat=false)
    end

    for precon in (:I, :diag, :chol, :ichol)
        (fhat, _, _) =
            b0map(finit, ydata, etime; niter=5, l2b=-3, track=false, precon, chat=false)
    end

jim(:prompt, true)
jif = (args...; kwargs...) -> jim(args...; prompt=false, kwargs...)

    for gamma_type in (:FR, :PR)
        (fhat, _, _) =
            b0map(finit, ydata, etime; niter=15, l2b=-4, mask, track=false, gamma_type, chat=false)
 
        @show maximum(abs, (fhat - ftrue) .* mask) #< 5
        jim(jif(ftrue), jif(finit.*mask, "finit"), jif(mask), jif(fhat), jif(fhat-ftrue), jif(xtrue, "true $gamma_type"))
#=
throw()
=#
    end

    # water-fat
    (fhat, _, out) =
        b0map(finit, ydata, etime; niter=5, l2b=-3, mask, df=500)
    @test maximum(abs, (fhat - ftrue) .* mask) < 8
    @test fhat isa Matrix{Float32}
    @test out.xw isa Matrix{ComplexF32} # todo: test better
    @test out.xf isa Matrix{ComplexF32}

    # multiple coil case
    smap = cat(fill(1f0+2im, ig.dims), fill(3-1im, ig.dims); dims=3) # (dims..., nc)

    ydata = [xtrue .* cis.(2f0π * ftrue * ΔTE) for ΔTE in etime]
    ydata = cat(dims=3, ydata...) # (dims..., ne)
#   ydata = ydata .* reshape(smap, ig.dims..., 1, :)
    ydata = reshape(ydata, ig.dims..., 1, :) # (dims..., 1, ne)
    ydata = smap .* ydata
    seed!(0)
    ydata .+= 0.04 * randn(ComplexF32, size(ydata))
    finit = angle.(ydata[:,:,1,2] .* conj(ydata[:,:,1,1])) / (etime[2] - etime[1]) / 2f0π

    (fhat, times, out) =
        b0map(finit, ydata, etime; smap, niter=9, l2b=-2, mask, chat=false)
    @test maximum(abs, (fhat - ftrue) .* mask) < 8
end

#=
using MIRTjim: jim
jim(axes(ig), ftrue)
jim(axes(ig), xtrue)
jim(angle.(ydata))
jim(jim(ftrue), jim(finit.*mask), jim(mask), jim(fhat), jim(fhat-ftrue), jim(xtrue))
=#
