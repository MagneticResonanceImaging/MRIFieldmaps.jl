#---------------------------------------------------------
# # [Fat-Water separation](@id 03-fat-water)
#---------------------------------------------------------

#=
This page illustrates regularized B0 field map estimation
in the context of fat-water separation
from multi-echo multi-coil MRI images
using the Julia package
[`MRIFieldmaps`](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).

This page was generated from a single Julia file:
[03-fat-water.jl](@__REPO_ROOT_URL__/03-fat-water.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`03-fat-water.ipynb`](@__NBVIEWER_ROOT_URL__/03-fat-water.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`03-fat-water.ipynb`](@__BINDER_ROOT_URL__/03-fat-water.ipynb).


# ### Setup

# Packages needed here.

using MRIFieldmaps: b0map, b0model, b0init, b0scale
using MIRTjim: jim, prompt; jim(:prompt, true)
using MAT: matread
import Downloads # todo: use Fetch or DataDeps?
using MIRT: ir_mri_sensemap_sim
using Random: seed!
using StatsBase: mean
#using Unitful: s
using Plots; default(markerstrokecolor=:auto, label="")
jif = (args...; kwargs...) -> jim(args...; prompt=false, kwargs...);


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
### Overview

This example is based on the simulation example from
[the repo](https://github.com/ClaireYLin/regularized-field-map-estimation)
that reproduces Experiment A and Figs. 7-9
in
[the paper](http://doi.org/10.1109/TCI.2020.3031082)
"Efficient Regularized Field Map Estimation in 3D MRI"
by Claire Lin and Jeff Fessler, 2020

This example is based on translating
[this matlab code](https://github.com/ClaireYLin/regularized-field-map-estimation/blob/main/examples/example_simu_waterfat.m)

This example uses units (seconds)
to illustrate that capability of the package,
but units are not required.
=#


#=
## Read data

Original data source:
https://www.ismrm.org/workshops/FatWater12/data.htm
=#
if !@isdefined(ydata)
    url = "https://github.com/JeffFessler/scratch/blob/main/linsim2.mat?raw=true" # todo
    data = matread(Downloads.download(url))
#src gdrive_folder = "https://drive.google.com/drive/folders/1SpeSIi_h7ulI-gLVpnBit89J4kH40nxs"
#src gdrive_file = "https://drive.google.com/file/d/1Sv_D0AmPRbjHT1a7oq7evwjNmf7Tyjff"
#src local_file = "linsim2.mat"
#src drive_download(gdrive_file, local_file) # fails
#src data = matread(local_file)

    df = vec(data["df"])
    relamp = vec(data["relamp"])
    mask = data["mask"]
    finit0 = data["finit0"]
    finit1 = data["finit1"]
    finit2 = data["finit2"] # after graph cut method
    ftrue = data["ftrue"]
    water_true = data["water_true"]
    fat_true = data["fat_true"]
    echotime = vec(data["echotime"])
    scale = data["scale"]
    ydata = data["ydata"]
    ydata = selectdim(ydata, 3, 1) # xycl = 256, 192, 1, 8)
    ytrue = data["ytrue"]
    ytrue = selectdim(ytrue, 3, 1) # ""
    @assert abs(norm(ydata)/norm(ytrue)*scale - 1) < 0.003 # check scale
end;


# Original image data
ne = length(echotime)
jim(data["images_original"], "Original |images| for $ne echo times"; ncol=3)


#=
From those images,
the following water, fat and field maps
were computed
using a golden-section search
to define "ground truth" images
for this simulation.
=#
jim(
 jif(water_true, "True water component"),
 jif(fat_true, "True fat component"),
 jif(ftrue, "True B0 map [Hz] (Fig. 8a)"),
 jif(mask, "Mask"),
)


#=
Verify that the simulated image data in Matlab
matches the model used here.

Because `b0model` uses `cis(+phase)`,
the resulting fieldmap
may be the negative of the needed for your scanner!
=#
tmp = b0model(ftrue, water_true, echotime; df, relamp, xfat=fat_true)
@assert maximum(abs, tmp - ytrue) / maximum(abs, ytrue) < 2f-6
jim(ydata, "Noisy |data| for $ne echo times"; ncol=3)


# Rescale data
# Not needed here because data was already pre-scaled in Matlab
#src ydata_scaled, scale = b0scale(ydata, echotime); # todo: move into b0map


# Function for computing RMSE within the mask
frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)), digits=1);


#=
## Initial fieldmap estimate

Compute `finit` via
a discrete maximum-likelihood search.
(no smoothing).
=#


# Verify Julia vs Matlab initialization.
if false
    fband = maximum(abs, df) # bandwidth that matches (dubious?) Matlab version
    finit = b0init(ydata, echotime; df, relamp, nf=100, fband, threshold=0)
    finit .*= mask
    @assert finit ≈ finit0
end

#=
Next the Matlab code runs
a regularized weighted-least squares method "PWLS"
to get a smooth initial fieldmap.
This is not yet implemented in Julia.

todo: revisit after better factoring the CG routine

For future reference:
`w1 = winit_pwls_pcg_ls(w0(mask), yik_c(mask,:,:), p.te,
        smap_c(mask,:),'order', 2, 'l2b', -1,
        'niter', 10,'maskR', mask,'precon','ichol',
        'gammaType','PR','df',df,'relamp',relAmps)`
=#

#=
Finally the Matlab version
sets background fieldmap pixels to mean of "good" pixels,
using a small threshold.

We emulate that approach here using `b0init`.
Here we use the built-in defaults
in `b0init`
for `fband`, `nf` and `threshold`. todo
=#

if !@isdefined(finit_julia)
    finit_julia = b0init(ydata, echotime; df, relamp, threshold=0.01)
end
finit = finit_julia .* mask;


# Uncomment the following line
# to use the Matlab initial field map
# for trying to reproduce the original plots.
finit = finit2;


# Show initial fieldmap
clim = (-100,200)
jie = (f,t) -> jif(f,t; clim, xlabel = "RMSE: $(frmse(f)) Hz")
jim(
 jif(ftrue, "ftrue"; clim),
 jie(finit0, "finit0 Fig 7b"),
 jie(finit2, "finit2 Fig 7c"),
 jie(finit, "finit"),
 jif(finit2-finit, "f2-fi"),
)


#=
## Graphcut implementation

Not done here

`algoParams.DO_OT = 0; # Optimal transfer
algoParams.OT_ITERS = 100; # number of optimal transfer iterations
algoParams.lambda = 2^(-7);# Regularization parameter
algoParams.LMAP_EXTRA = 0; # More smoothing for low-signal regions
algoParams.LMAP_POWER = 0; # Spatially-varying regularization (2 gives ~ uniform resolution)
algoParams.NUM_ITERS = 100; # Number of graph cut iterations
imDataParams.finit = winit/2/pi;
[outParams,time_gc] = fmap_est_graphcut(imDataParams, algoParams);
wmap_gc = 2*pi*outParams.fms.*mask;
argsError_gc = {'GC', time_gc, wmap_gc};`
=#


#=
## Run NCG

Run each algorithm twice;
once to track RMSE and costs,
once for timing.

Caution:
note the need to correct for the data scale factor.
=#
order, l2b = 1, -4 # todo
order, l2b = 2, -7 # from Matlab version
fmap_run = (niter, precon, track; kwargs...) ->
    b0map(ydata, echotime; mask, df, relamp, finit,
       order, l2b, gamma_type=:PR, niter, precon, track, kwargs...)

function runner(niter, precon; kwargs...)
    (fmap, _, out) = fmap_run(niter, precon, true; kwargs...) # tracking run
    (_, times, _) = fmap_run(niter, precon, false; kwargs...) # timing run
    return (fmap, out.fhats, out.costs, times, scale*out.xw, scale*out.xf)
end;


# ### 2. NCG: no precon
if !@isdefined(fmap_cg_n) || true
    niter_cg_n = 50
    (fmap_cg_n, fhat_cg_n, cost_cg_n, time_cg_n, xw_n, xf_n) = runner(niter_cg_n, :I)

    pcost = plot(time_cg_n, cost_cg_n, marker=:circle, label="NCG-MLS");
    pi_cn = jim(fmap_cg_n, "CG:I"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_n)) Hz")
end


# ### 3. NCG: diagonal preconditioner
if !@isdefined(fmap_cg_d) || true
    niter_cg_d = 60
    (fmap_cg_d, fhat_cg_d, cost_cg_d, time_cg_d, xw_d, xf_d) = runner(niter_cg_d, :diag)

    plot!(pcost, time_cg_d, cost_cg_d, marker=:square, label="NCG-MLS-D")
    pi_cd = jim(fmap_cg_d, "CG:diag"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_d)) Hz")
end


# ### 4. NCG: Cholesky preconditioner
# (This one may use too much memory for larger images.)
if !@isdefined(fmap_cg_c) || true
if false
    niter_cg_c = 15
    (fmap_cg_c, fhat_cg_c, cost_cg_c, time_cg_c, xw_c, xf_c) = runner(niter_cg_c, :chol)

    plot!(pcost, time_cg_c, cost_cg_c, marker=:square, label="NCG-MLS-C")
    pi_cc = jim(fmap_cg_c, "CG:chol"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_c)) Hz")
end
end


# ### 5. NCG: Incomplete Cholesky preconditioner
if !@isdefined(fmap_cg_i) || true
    niter_cg_i = 40
    (fmap_cg_i, fhat_cg_i, cost_cg_i, time_cg_i, xw_i, xf_i) =
        runner(niter_cg_i, :ichol; lldl_args = (; memory=20, droptol=0))

    plot!(pcost, time_cg_i, cost_cg_i, marker=:square, label="NCG-MLS-IC",
        xlabel = "time [s]", ylabel="cost")
    pi_ci = jim(fmap_cg_i, "CG:ichol"; clim,
        xlabel = "RMSE = $(frmse(fmap_cg_i)) Hz")
end


# Compare final RMSE values
frmse.((ftrue, finit, fmap_cg_n, fmap_cg_d, #=fmap_cg_c,=# fmap_cg_i))

# Plot RMSE vs wall time
prmse = plot(xlabel = "time [s]", ylabel="RMSE [Hz]", ylims=(0,100))
fun = (time, fhat, label) ->
    plot!(prmse, time, frmse.(eachslice(fhat; dims=3)); label, marker=:circ)
fun(time_cg_n, fhat_cg_n, "None")
fun(time_cg_d, fhat_cg_d, "Diag")
fun(time_cg_c, fhat_cg_c, "Chol")
fun(time_cg_i, fhat_cg_i, "IC")


# Estimated field map and errors
phat = jim(
 jif(ftrue, "ftrue"; clim),
 jif(mask, "mask"),
 jie(finit, "finit"),
 jie(fmap_cg_n, "None"),
 jie(fmap_cg_d, "Diag"),
 jie(fmap_cg_i, "IC"),
)


# Estimated water map and errors
wlim = (0, 100)
pwater = jim(
 jif(water_true, "Water true"; clim=wlim),
 jif(mask, "mask"),
 jif(xw_i, "Water IC"; clim=wlim),
 jif(abs.(xw_i - water_true), "|Error|"),
)


# Estimated fat map and errors
flim = (0, 100)
pfat = jim(
 jif(fat_true, "Fat true"; clim=flim),
 jif(mask, "mask"),
 jif(xf_i, "Fat IC"; clim=flim),
 jif(abs.(xf_i - fat_true), "|Error|"),
)


#=
## Discussion

The errors are similar to those in the 2020 paper,
but still seem larger than expected.
Further investigation is needed.
todo: probably a scale-factor issue!

The errors are similar even when initializing with `ftrue`.

The RMSE figure is somewhat similar to Fig. 9 of the 2020 Lin&Fessler paper.

This figure was generated in github's cloud,
where the servers are busily multi-tasking,
so the compute times per iteration
can vary widely between iterations and runs.

Here the incomplete Cholesky preconditioner
was very effective
relative to the diagonal preconditioner.

The Cholesky preconditioner also works well here,
but this is only a 2D problem.
For a 3D fieldmap,
it would likely require too much memory.
=#
