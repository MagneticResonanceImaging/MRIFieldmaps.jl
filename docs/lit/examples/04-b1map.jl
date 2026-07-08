#=
# [B1+ mapping](@id 04-b1map)

This page illustrates regularized B1+ map estimation
from MRI images
using the Julia package
[`MRIFieldmaps`](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).
=#

#srcURL


# ### Setup

# Packages needed here.

using ImagePhantoms: ellipse_parameters, SheppLoganBrainWeb, ellipse
using ImagePhantoms: phantom
#src using MRIFieldmaps: #todo
using MIRTjim: jim, prompt # todo ; jim(:prompt, true)
#src using MAT: matread
#src import Downloads # todo: use Fetch or DataDeps?
#src using MIRT: ir_mri_sensemap_sim
using Random: seed!
#src using StatsBase: mean
using Unitful: mm
using Plots: default; default(markerstrokecolor=:auto, label="")


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Overview

The approach considered here
is based on methods described
in Ch.~V of the
[2011 PhD Thesis of Amanda Funai](https://hdl.handle.net/2027.42/86473).
That work in term is a significant extension
of the 2007 ISBI paper
["Regularized B1+ map estimation in MRI"](https://doi.org/10.1109/ISBI.2007.356927)
by Amanda Funai, J A Fessler, W Grissom, D C Noll.

Note that strictly speaking B₁ has units of Tesla (magnetic field strength),
but in this work we are really just mapping
a scaling factor κ
that is the ratio
of the apparent B₁ divided by the target B₁ value,
i.e.,
if we prescribe a 90%
=#

#=
## Simulate data
For simplicity we consider simulated data here
using a basic Shepp-Logan type of ellipse phantom
and highly simplified transmit and receive coils.
=#

# Image geometry:

fovs = (256mm, 250mm)
nx, ny = (128, 100) .* 2
dx, dy = fovs ./ (nx,ny)
x = (-(nx÷2):(nx÷2-1)) * dx
y = (-(ny÷2):(ny÷2-1)) * dy;

#=
Define Shepp-Logan phantom object,
with random complex phases
per inner ellipse
to make it a bit more realistic.
=#

params = ellipse_parameters(SheppLoganBrainWeb() ; disjoint=true, fovs)
seed!(0)
phases = [1; rand(ComplexF32,9)] # random phases
params = [(p[1:5]..., phases[i]) for (i, p) in enumerate(params)]
oa = ellipse(params)
oversample = 3
image0 = phantom(x, y, oa, oversample)
ri_fun = z -> cat(dims = ndims(z)+1, real(z), imag(z))
p0 = jim(x, y, ri_fun(image0), "Digital phantom\n (real | imag)")

#=
In practice, sensitivity maps are usually estimated
only over portion of the image array,
so we define a simple `mask` here
to exercise this issue.
=#

mask = trues(nx,ny)
mask[:,[1:2;end-2:end]] .= false # mask out outer border of maps
mask[[1:8;end-8:end],:] .= false
@assert mask .* image0 == image0
pmask = jim(x, y, mask, "mask"; xlabel="x", ylabel="y")


#=
## Sensitivity maps (coil receive)

Here we use 2 highly idealized receive coil sensitivity maps,
roughly corresponding to the
[Biot-Savart law](https://en.wikipedia.org/wiki/Biot-Savart_law)
for an infinite thin wire,
as a crude approximation of a
[birdcage coil](https://en.wikipedia.org/wiki/Radiofrequency_coil).
One wire is outside the upper right corner,
the other is outside the left border.
=#

"""
    biot_savart_wire(x, y, wx, wy)
Compute response at `(x,y)` to wire at `(wx,wy)`
"""
function biot_savart_wire(x, y, wx, wy)
    phase = cis(atan(y-wy, x-wx))
    return oneunit(x) / sqrt(sum(abs2, (x-wx, y-wy))) * phase # 1/r falloff
end

function _rcv(wx, wy)
    wire = (a,b) -> biot_savart_wire(a, b, wx, wy)
#src smap[1] *= cis(3π/4) # match coil phases at image center, ala "quadrature phase"
    return wire.(x, y')
end

smap = stack(splat(_rcv), (
    (maximum(x) + 8dx, maximum(y) + 8dy),
    (maximum(x) + 8dx, minimum(y) - 8dy),
    (minimum(x) - 20dx, 0dy),
  )
)

#=
Typical sensitivity map estimation methods
normalize the maps
so that the square-root of the sum of squares (SSoS) is unity:
=#
ssos = sqrt.(sum(abs2, smap, dims=ndims(smap))) # SSoS
ssos = selectdim(ssos, ndims(smap), 1)

ncoilr = last(size(smap))
for ic in 1:ncoilr # normalize
    selectdim(smap, ndims(smap), ic) ./= ssos
end
smap .*= mask
smaps = collect(eachslice(smap, dims=3)) # code hereafter expects vector of maps

ps = jim(
 jim(x, y, abs.(smap), " |Sensitivity maps raw| (ncoilr=$ncoilr)";
  color=:cividis, ncol=ncoilr, prompt=false),
 jim(x, y, angle.(smap), "∠(Sensitivity maps raw)";
  color=:hsv, ncol=ncoilr, prompt=false,
  clim=(-π,π), colorbar_ticks = ([-π, 0, π], ["-π", "0", "π"]),
 ),
 layout = (2,1),
)


#=
## B1+ transmit maps: ground truth
=#
ncoilt = 6
function _xmit(it::Int)
    wx, wy = sincos(2π/ncoilt*it) .* 3maximum(x)
    wire = (a,b) -> biot_savart_wire(a, b, wx, wy)
    wire.(x, y')
end
tmp = map(_xmit, 1:ncoilt)
tmp = cat(dims=3, tmp...)
#src b1map_true = tmp / maximum(abs, tmp)
b1map_true = tmp / abs(tmp[end÷2,end÷2,1]) # ≈1 at middle

p1 = jim(
 jim(x, y, abs.(b1map_true); ncol=ncoilt, title="Magnitude",
  prompt=false, xlabel="x", ylabel="y"),
 jim(x, y, angle.(b1map_true); ncol=ncoilt, title="Phase", color=:hsv,
  prompt=false, clim=(-π,π), colorbar_ticks = ([-π, 0, π], ["-π", "0", "π"])),
 plot_title="$ncoilt |B1+| maps: Ground truth",
 layout = (2,1),
)


#=
## One-at-a-time double angle measurement
=#
αtarget = 2π/3 # target flip angle
chi = [1; 2] * αtarget
M = size(chi,1) # number of measurements
Hfun = sin # ideal model that ignores steady-state and slice-profile effects
Ffun(z) = sign(z) .* Hfun(z)
N = (nx, ny) # image dimensions
xtrue = reshape(chi, 1, 1, 1, M) .*
        reshape(b1map_true, N..., ncoilt, 1) # (N) × ncoilt × M
ρtrue = image0 .* Hfun.(xtrue) # excited magnetization (N) × ncoilt × M
ytrue = reshape(smap, N..., 1, 1, ncoilr) .*
        reshape(ρtrue, N..., ncoilt, M, 1) # (N) × ncoilt × M × ncoilr

# Add noise
σ = 0.1 # small noise for now
ymeas = ytrue + σ * randn(ComplexF32, size(ytrue))

#=
Receive coil complex combination
using ideal receive coil sensitivity maps
for simplicity.
=#
ycomb = conj(reshape(smap, N..., 1, 1, ncoilr)) .* ymeas
ycomb = sum(ycomb, dims = ndims(ycomb))
ycomb = reshape(ycomb, N..., ncoilt, M) # (N) × ncoilt × M

jim(x, y, abs.(ycomb); nrow=M, ncol=ncoilt,
 title = "Coil-combined data for M=$M")


throw() # xx todo

smap = [wire1.(x, y'), wire2.(x, y')]
smap[1] *= cis(3π/4) # match coil phases at image center, ala "quadrature phase"
#src mag = abs.(smap)
#src phase = angle.(smap)

# Extract arrays used in simulation
if !@isdefined(ftrue)
    zp = 1:40 # choose subset of slices
    mask = data["maskR"][:,:,zp]
    ftrue = (data["in_obj"]["ztrue"][:,:,zp] .* mask) / 2π / 1s # Hz
    ftrue .*= mask # true field map (in Hz) for simulation
    mag = data["in_obj"]["xtrue"] .* mask # true baseline magnitude
    if false # 2× in all 3 dimensions to make (128,128,80) for timing test (≈6sec)
        catd = (x,d) -> cat(x, x, dims=d)
        bigify = (x) -> catd(catd(catd(x, 1), 2), 3)
        ftrue = bigify(ftrue)
        mag = bigify(mag)
        mask = bigify(mask)
    end
    (nx,ny,nz) = size(mag)
    clim = (-100,100) # display range in Hz
    jim(ftrue .* mask; clim, title="True fieldmap in Hz (Fig 3d)")
end

# Function for computing RMSE within the mask
frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)) * s, digits=1) / s;


# Parameters for data generation
echotime = [0, 2, 10] * 1f-3 * 1s # echo times in sec
true_thresh = 0.05 # threshold of the true object for determining reconstruction mask
snr = 24 # noise level in dB
ne = length(echotime)
nc = 4; # number of coils in simulation

# ## Simulate sensitivity maps
# (`rcoil=100` to match matlab default).
# todo: polynomial approximation?
if !@isdefined(smap)
    smap = ir_mri_sensemap_sim(; dims=(nx, ny, nz), ncoil=nc, rcoil=100)
    div0 = (x::Number,y::Number) -> iszero(y) ? 0 : x/y
    smap ./= sqrt.(sum(abs2, smap; dims=4)) # normalize by SSoS
    jim(smap, "|smap|"; ncol=nz÷2)
end


#=
## Generate simulated image data
This is the multi-coil version,
for multiple echo times,
with additive complex Gaussian noise.

Because `b0model` uses `cis(+phase)`,
the resulting fieldmap
may be the negative of the needed for your scanner!
=#
ytrue = b0model(ftrue, mag, echotime; smap)
seed!(0) # matlab and julia will differ
 # compute the noise_std to get the desired SNR
image_power = 10 * log10(sum(abs2, mag) / (nx*ny*nz)) # in dB
noise_power = image_power - snr
noise_std = sqrt(10^(noise_power/10)) / 2 # because complex
ynoise = Float32(noise_std) * randn(ComplexF32, size(ytrue))
ydata = ytrue + ynoise; # add the noise to the data
# Compute the SNR for each echo time to verify
tmp = [sum(abs2, ytrue[:,:,:,:,i]) / sum(abs2, ynoise[:,:,:,:,i]) for i in 1:ne]
datasnr = 10 * log10.(tmp)

# Show data magnitude
jim(ydata[:,:,:,:,end], "|data|"; ncol=nz÷2)


# Coil combine image data and scale
if !@isdefined(yik_sos)
    yik_sos = sum(conj(smap) .* ydata; dims=4) # coil combine
    yik_sos = yik_sos[:,:,:,1,:] # (dims..., ne)
    jim(yik_sos, "|data sos|"; ncol=nz÷2)
    (yik_sos_scaled, scale) = b0scale(yik_sos, echotime) # todo
    jim(yik_sos_scaled, "|scaled data|"; ncol=nz÷2)
end


#=
## Initialize fieldmap

Compute `finit`
using phase difference of first two echo times (no smoothing):
=#
finit = b0init(ydata, echotime; smap)
jim(finit .* mask; clim, title="Initial fieldmap in Hz (Fig 3b)",
    xlabel = "RMSE = $(frmse(finit)) Hz")

#src # QM-Huber / NCG for 3D fieldmap estimation

#=
## Run NCG

Run each algorithm twice; once to track rmse and costs, once for timing
=#
yik_scale = ydata / scale
fmap_run = (niter, precon, track; kwargs...) ->
    b0map(yik_scale, echotime; smap, mask,
       order=1, l2b=-4, gamma_type=:PR, niter, precon, track, kwargs...)

function runner(niter, precon; kwargs...)
    (fmap, _, out) = fmap_run(niter, precon, true; kwargs...) # tracking run
    (_, times, _) = fmap_run(niter, precon, false; kwargs...) # timing run
    return (fmap, out.fhats, out.costs, times)
end;


# ### 2. NCG: no precon
if !@isdefined(fmap_cg_n)
    niter_cg_n = 50
    (fmap_cg_n, fhat_cg_n, cost_cg_n, time_cg_n) = runner(niter_cg_n, :I)

    pcost = plot(time_cg_n, cost_cg_n, marker=:circle, label="NCG-MLS");
    pi_cn = jim(fmap_cg_n, "CG:I"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_n)) Hz")
end


# ### 3. NCG: diagonal preconditioner
if !@isdefined(fmap_cg_d)
    niter_cg_d = 40
    (fmap_cg_d, fhat_cg_d, cost_cg_d, time_cg_d) = runner(niter_cg_d, :diag)

    plot!(pcost, time_cg_d, cost_cg_d, marker=:square, label="NCG-MLS-D")
    pi_cd = jim(fmap_cg_d, "CG:diag"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_d)) Hz")
end


# ### 4. NCG: Cholesky preconditioner
# (This one may use too much memory for larger images.)
if !@isdefined(fmap_cg_c)
    niter_cg_c = 3
    (fmap_cg_c, fhat_cg_c, cost_cg_c, time_cg_c) = runner(niter_cg_c, :chol)

    plot!(pcost, time_cg_c, cost_cg_c, marker=:square, label="NCG-MLS-C")
    pi_cc = jim(fmap_cg_c, "CG:chol"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_c)) Hz")
end


# ### 5. NCG: Incomplete Cholesky preconditioner
if !@isdefined(fmap_cg_i)
    niter_cg_i = 14
    (fmap_cg_i, fhat_cg_i, cost_cg_i, time_cg_i) =
        runner(niter_cg_i, :ichol; lldl_args = (; memory=20, droptol=0))

    plot!(pcost, time_cg_i, cost_cg_i, marker=:square, label="NCG-MLS-IC",
        xlabel = "time [s]", ylabel="cost")
    pi_ci = jim(fmap_cg_i, "CG:ichol"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_i)) Hz")
end


# Compare final RMSE values
frmse.((ftrue, finit, fmap_cg_n, fmap_cg_d, fmap_cg_c, fmap_cg_i))

# Plot RMSE vs wall time
prmse = plot(xlabel = "time [s]", ylabel="RMSE [Hz]")
fun = (time, fhat, label) ->
    plot!(prmse, time, frmse.(eachslice(fhat; dims=4)); label, marker=:circ)
fun(time_cg_n, fhat_cg_n, "None")
fun(time_cg_d, fhat_cg_d, "Diag")
fun(time_cg_c, fhat_cg_c, "Chol")
fun(time_cg_i, fhat_cg_i, "IC")

#=
## Discussion

That final figure is similar to Fig. 4 of the 2020 Lin&Fessler paper,
after correcting that figure for a
[factor of π](https://github.com/ClaireYLin/regularized-field-map-estimation).

This figure was generated in github's cloud,
where the servers are busily multi-tasking,
so the compute times per iteration
can vary widely between iterations and runs.

Nevertheless,
it is interesting that
in this Julia implementation
the diagonal preconditioner
seems to be
as effective as the incomplete Cholesky preconditioner.
=#
