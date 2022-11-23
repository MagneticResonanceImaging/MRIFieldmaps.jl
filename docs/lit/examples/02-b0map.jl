#---------------------------------------------------------
# # [B0 field map](@id 02-b0map)
#---------------------------------------------------------

#=
This page illustrates regularized B0 3D field map estimation
from multi-echo multi-coil MRI images
using the Julia package
[`MRIFieldmaps`](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).

This page was generated from a single Julia file:
[02-b0map.jl](@__REPO_ROOT_URL__/02-b0map.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`02-b0map.ipynb`](@__NBVIEWER_ROOT_URL__/02-b0map.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`02-b0map.ipynb`](@__BINDER_ROOT_URL__/02-b0map.ipynb).


# ### Setup

# Packages needed here.

using MRIFieldmaps: b0map, b0model, b0init, b0scale
using MIRTjim: jim, prompt; jim(:prompt, true)
using MAT: matread
import Downloads # todo: use Fetch or DataDeps?
using MIRT: ir_mri_sensemap_sim
using Random: seed!
using StatsBase: mean
using Unitful: s
using Plots; default(markerstrokecolor=:auto, label="")


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
### Overview

This example is based on the simulation example from
[the repo](https://github.com/ClaireYLin/regularized-field-map-estimation)
that reproduces Experiment A and Figs. 3 & 4
in
[the paper](http://doi.org/10.1109/TCI.2020.3031082)
"Efficient Regularized Field Map Estimation in 3D MRI"
by Claire Lin and Jeff Fessler, 2020

This example uses units (seconds)
to illustrate that capability of the package,
but units are not required.
=#

# ## Read data
if !@isdefined(data)
    repo = "https://github.com/ClaireYLin/regularized-field-map-estimation"
    dataurl = "$repo/blob/main/data/input_object_40sl_3d_epi_snr40.mat?raw=true"
    data = matread(Downloads.download(dataurl))
end;

# Extract arrays used in simulation
if !@isdefined(ftrue)
    zp = 1:40 # choose subset of slices
    mask = data["maskR"][:,:,zp]
    ftrue = (data["in_obj"]["ztrue"][:,:,zp] .* mask) / 2π / 1s # Hz
    ftrue .*= mask # true field map (in Hz) for simulation
    mag = data["in_obj"]["xtrue"] .* mask # true baseline magnitude
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
