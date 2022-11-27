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

This example uses units (seconds)
to illustrate that capability of the package,
but units are not required.
=#

#src https://github.com/ClaireYLin/regularized-field-map-estimation/blob/main/examples/example_simu_waterfat.m

#src using GoogleDrive: drive_download # todo fails

# ## Read data
# Original data source: https://www.ismrm.org/workshops/FatWater12/data.htm
if !@isdefined(ydata)
    url = "https://github.com/JeffFessler/scratch/blob/main/linsim2.mat?raw=true"
#   gdrive_folder = "https://drive.google.com/drive/folders/1SpeSIi_h7ulI-gLVpnBit89J4kH40nxs"
#   gdrive_file = "https://drive.google.com/file/d/1Sv_D0AmPRbjHT1a7oq7evwjNmf7Tyjff"
#   local_file = "linsim2.mat"
#src drive_download(gdrive_file, local_file) # fails
    data = matread(Downloads.download(url))
#   data = matread(local_file)

    df = vec(data["df"])
    relamp = vec(data["df"])
    mask = data["mask"]
    finit0 = data["finit0"]
    finit1 = data["finit1"]
    finit2 = data["finit2"] # after graph cut todo!
    ftrue = data["ftrue"]
    water_true = data["water_true"]
    fat_true = data["fat_true"]
    echotime = vec(data["echotime"])
    ydata = data["yik"]
    ydata = selectdim(ydata, 3, 1) # xycl = 256, 192, 1, 8)
end;

# plot images
#jim(imwater)
#jim(imfat)
#jim(ftrue)
#    jim(ftrue .* mask; clim, title="True fieldmap in Hz (Fig 3d)")
#jim(ydata[:,:,:,:,end], "|data|"; ncol=nz÷2)

# todo: compare b0model to ytrue?

#=
## Generate simulated image data
This is the multi-coil version,
for multiple echo times,
with additive complex Gaussian noise.

Because `b0model` uses `cis(+phase)`,
the resulting fieldmap
may be the negative of the needed for your scanner!
=#
# ytrue = b0model(ftrue, mag, echotime)



# rescale
ydata_scaled, scale = b0scale(ydata, echotime)

#=
Initialize fieldmap 2: PWLS (todo?)
printm(' -- PWLS on winit -- ')
y = reshape(yik,[],nc,ne);
w1 = winit_pwls_pcg_ls(w0(mask), yik_c(mask,:,:), p.te,...
        smap_c(mask,:),'order', 2, 'l2b', -1,  ...
        'niter', 10,'maskR', mask,'precon','ichol',...
        'gammaType','PR','df',df,'relamp',relAmps);
# Initialize fieldmap 3: set background pixels to mean of "good" pixels
winit = w1;
p.yk_thresh = 0.01;
good = winit > p.yk_thresh * max(yik(:));
winit(~good) = mean(winit(good));
# plot winit
winit = reshape(winit,[nx,ny,nz]);
=#

#=
# graphcut?
## 0. Graphcut implementation
algoParams.DO_OT = 0; # Optimal transfer
algoParams.OT_ITERS = 100; # number of optimal transfer iterations
algoParams.lambda = 2^(-7);# Regularization parameter
algoParams.LMAP_EXTRA = 0; # More smoothing for low-signal regions
algoParams.LMAP_POWER = 0; # Spatially-varying regularization (2 gives ~ uniformn resolution)
algoParams.NUM_ITERS = 100; # Number of graph cut iterations
imDataParams.finit = winit/2/pi;
[outParams,time_gc] = fmap_est_graphcut(imDataParams, algoParams);
wmap_gc = 2*pi*outParams.fms.*mask;
argsError_gc = {'GC', time_gc, wmap_gc};
=#

#=
## 2. NCG implementation: ichol precon
niter_cg = 200;
[out,cost_cg,time_cg] = fmap_est_pcg_ls(winit(mask), yik_c(mask,:,:),p.te, ...
    smap_c(mask,:),'order', 2, 'l2b', -7, ...
    'niter', niter_cg,'maskR', mask,'precon','ichol',...
    'gammaType','PR','df',df,'relamp',relAmps);
wmap_cg = embed(out.ws,mask);
imwater_cg = embed(out.xw,mask);
imfat_cg = embed(out.xf,mask);
figure(2);subplot(121);im(wmap_cg(:,:,end), 'cg', wlim)
subplot(122);semilogy(time_cg,cost_cg,'.-k')
argsError_cg = {'NCG-MLS-IC', time_cg, wmap_cg};
## RMSE plots
argsError = {argsError_qm{:};argsError_gc{:}; argsError_cg{:}};
error = compute_rmsd(argsError, wtrue,'step',20);
# labels
grid on
axis([0,100,0,60])
xlabel('Time (s)')
ylabel('RMSE (Hz)')
=#


# # NOT DONE BELOW HERE todo

# Function for computing RMSE within the mask
# frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)) * s, digits=1) / s;
frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)), digits=1);

# Parameters for data generation
ne = length(echotime)
nc = 1; # number of coils in simulation


#=
## Initialize fieldmap

Compute `finit`
using phase difference of first two echo times (no smoothing):
=#

clim = (-100,200)
if false
    finit = b0init(ydata, echotime; df, relamp) # not so great!
    fband = maximum(abs, df) # to match claire lin's code
    finit3 = b0init(ydata, echotime; df, relamp, nf=100, fband) # not great either
jim(
 jim(ftrue; clim),
 jim(finit; clim),
 jim(finit0; clim),
#jim(finit1; clim),
 jim(finit2; clim),
)

jim(
 jim(finit2, "finit2"; clim),
 jim(finit3, "finit3"; clim),
)
end
finit = finit2 # todo for now
jim(finit .* mask; clim, title="Initial fieldmap in Hz (Fig 3b)",
    xlabel = "RMSE = $(frmse(finit)) Hz")


#=
## Run NCG

Run each algorithm twice; once to track rmse and costs, once for timing
=#
yik_scale = ydata / scale
fmap_run = (niter, precon, track; kwargs...) ->
    b0map(yik_scale, echotime; mask, df, relamp,
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

throw()

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
