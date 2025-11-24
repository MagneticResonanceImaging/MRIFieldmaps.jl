#=
# [Phase contrast B0 mapping](@id 03-phasecontrast)

This page illustrates regularized B0 3D field map estimation
from multi-echo, multi-coil MRI images
using the Julia package
[`MRIFieldmaps`](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).
In this case,
we assume we do not have access to coil sensitivity maps.
Instead of using those maps for coil combination,
we will use a phase contrast-based approach.
=#

#srcURL


# ### Setup

# Packages needed here.

using MRIFieldmaps: b0map, phasecontrast
using MIRTjim: jim, prompt; jim(:prompt, true)
using Random: seed!
using Unitful: s, ms, Hz


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Overview

When estimating B0 field maps
from multi-coil data,
the images first can be coil-combined
in a way that preserves image phase.
Ideally,
one has access to coil sensitivity maps
that can be used
for the coil combination.
However,
coil sensitivity maps
are not always available
(e.g., for B0 shimming).
One way of coil combining the images
without coil sensitivity maps
is to pretend all the coils
are uniformly sensitive
across the object
(i.e., assume the sensitivity maps
are uniformly equal to 1).
Alternatively,
one can effectively use the multi-coil images themselves
to glean information
about the coil sensitivities
and coil combine the images
using a phase contrast-based approach.

We now review the B0 mapping procedure
so that we can then understand
how the choice of coil combination scheme
influences the estimated field map.
For the case we are considering,
where coil combination is done
as a preprocessing step,
the regularized, iterative B0 mapping procedure
seeks to minimize
```math
\Psi(\mathbf{\omega}) = \Phi(\mathbf{\omega}) + \frac{\beta}{2} \|\mathbf{C} \mathbf{\omega}\|_2^2,
```
where $\mathbf{\omega}$ is the field map,
$\Phi(\mathbf{\omega})$ computes the data fit term,
$\mathbf{C}$ is a finite difference operator
to encourage spatial smoothness
of the field map,
and $\beta$ is a regularization parameter.
The data fit term is given by
```math
\Phi(\mathbf{\omega}) = \sum_{j = 1}^{N_{\mathrm{v}}} \sum_{m, n = 1}^{N_{\mathrm{e}}} \phi_{mnj}(\omega_j),
```
where $N_{\mathrm{v}}$ is the number of image voxels,
$N_{\mathrm{e}}$ is the number of echoes,
and
```math
\phi_{mnj}(\omega_j) = |r_{mnj}| [1 - \cos(\angle r_{mnj} + \omega_j (t_m - t_n))].
```
Here,
$t_m - t_n$ represents the echo time difference
between the $m$th and $n$th echoes,
and
```math
r_{mnj} = \frac{1}{N_{\mathrm{e}}} z_{mj}^{*} z_{nj},
```
where $z_{nj}$ is the (coil-combined) image value
for echo $n$ at voxel $j$,
and $(\cdot)^{*}$ denotes complex conjugate.

We now consider what $z_{nj}$ looks like
for the coil combination schemes discussed.

1. *Coil combination using coil sensitivities:*
   In this case,
   we first compute
   the sum-of-squares of the sensitivity maps:
   ```math
   v_j = \sum_{c = 1}^{N_{\mathrm{c}}} |s_{cj}|^2,
   ```
   where $s_{cj}$ is the coil sensitivity
   of coil $c$ at voxel $j$,
   and $N_{\mathrm{c}}$ is the number of coils.
   We then use the sensitivity maps
   for the coil combination:
   ```math
   z_{nj} = \frac{1}{\sqrt{v_j}} \sum_{c = 1}^{N_{\mathrm{c}}} s_{cj}^{*} y_{cnj},
   ```
   where $y_{cnj}$ is the image data
   for coil $c$ of echo $n$ at voxel $j$.
1. *Taking coil sensitivities to be uniformly equal to 1:*
   In this case,
   $s_{cj} = 1 \, \forall c, j$,
   so $v_j = N_{\mathrm{c}}$
   and
   ```math
   z_{nj} = \frac{1}{\sqrt{N_{\mathrm{c}}}} \sum_{c = 1}^{N_{\mathrm{c}}} y_{cnj}.
   ```
1. *Phase contrast-based approach:*
   In this case,
   we first coil combine the first echo image
   by taking the sum-of-squares
   across coils:
   ```math
   v_j = \sum_{c = 1}^{N_{\mathrm{c}}} |y_{c1j}|^2,
   ```
   where $y_{c1j}$ is the image data
   for coil $c$ of the first echo at voxel $j$.
   Then, for each echo,
   we coil combine in the following way:
   ```math
   z_{nj} = \frac{1}{\sqrt{v_j}} \sum_{c = 1}^{N_{\mathrm{c}}} y_{c1j}^{*} y_{cnj}.
   ```

The example that follows shows a simulated experiment
where a B0 field map is estimated
without knowledge
of the coil sensitivities.
This example examines B0 maps estimated
when using both coil combination approaches
and compares them to the case
where coil sensitivity information is known.
This example also compares
the above regularized, iteratively estimated B0 maps
to one obtained
using a (non-iterative) phase contrast B0 mapping approach.
=#

# ## Get simulated data

# ### Sensitivity maps

# Create a function for generating Gaussian-shaped, pure-real sensitivity maps.
function sensitivity_map(nx, ny, cx, cy, σ)

    s = [exp(-((x - cx)^2 + (y - cy)^2) / σ^2) for x in 1:nx, y in 1:ny]
    return complex.(s ./ maximum(s))

end;

# Specify the parameters for the sensitivity maps
# and then generate the maps.
nx = ny = 64
σ = 60
centers = [
    (nx ÷ 2, -100),
    (nx ÷ 2, 100 + ny + 1),
    (-100, ny ÷ 2),
    (100 + nx + 1, ny ÷ 2),
]
smap = cat([sensitivity_map(nx, ny, cx, cy, σ) for (cx, cy) in centers]...; dims = 3)
jim(smap; title = "Sensitivity maps")

# ### Object

# Create a disk-shaped object
# with uniform intensity (M0) throughout.
radius = 25
obj = [(x - nx÷2)^2 + (y - ny÷2)^2 < radius^2 for x in 1:nx, y in 1:ny]
mask = obj .> 0 # Mask indicating object support
jim(obj; title = "Object")

# ### True field map

# Create a linearly (spatially) varying field map.
ftrue = repeat(range(-50, 50, nx), 1, ny) * Hz
ftrue[nx÷2,ny÷2] = 50Hz # Add Kronecker impulse for visualizing regularization-induced blur
ftrue .*= mask # Mask out background voxels for better visualization
clim = (-60, 60) .* Hz # Use common colorbar limits for all field map plots
jim(ftrue; title = "True field map", clim)

# ### Data

# Specify parameters for the simulated multi-echo, multi-coil data.
TE = (4.0ms, 6.3ms)
T2 = 40ms
σ_noise = 0.005; # Noise standard deviation

# Make a function for creating the simulated data
# for a given echo time.
function make_data(TE)

    data = @. smap * obj * exp(-TE / T2) * cispi(2TE * ftrue) # Noiseless
    @. data += complex(σ_noise * randn(), σ_noise * randn()) # Add complex Gaussian noise
    return data

end;

# Create the simulated data.
seed!(0)
ydata = cat(make_data.(TE)...; dims = 4)
jim(ydata; title = "Simulated data", ncol = 4)


#=
## Estimate B0

Now that we have the simulated data,
we can estimate B0 from the data
using the different approaches.

First, create a function for computing the RMSE of the estimated B0 maps
and another function for displaying the B0 maps.
=#
frmsd = (f1, f2) -> 1Hz * round(sqrt(sum(abs2, (f1 - f2)[mask]) / count(mask))/1Hz; digits = 1)
frmse = f -> frmsd(f, ftrue)
plotb0 = f -> jim(f; title = "RMSE = $(frmse(f))", clim);

#=
Also specify the strength of the regularization parameter,
as well as the type of preconditioner to use.
(Note that the default preconditioner used by `b0map` (`precon = :ichol`)
does not produce good results in this example.)
=#
l2b = -26 # Regularization parameter is β = 2^l2b
precon = :diag;

# ### Use coil sensitivity maps

fcoil = b0map(ydata, TE; smap, l2b, precon)[1] .* mask
plotb0(fcoil)

# ### Assume uniform coil sensitivities

funiform = b0map(ydata, TE; smap = ones(eltype(smap), size(smap)), l2b, precon)[1] .* mask
plotb0(funiform)

# ### Use phase contrast-based approach

fpc_iterative = b0map(ydata, TE; smap = nothing, l2b, precon)[1] .* mask
plotb0(fpc_iterative)

# ### Do conventional (non-iterative) phase contrast B0 mapping

fpc = phasecontrast(ydata, TE) .* mask
plotb0(fpc)

# ### Show all field maps together

annotate = (col, row, title; fontsize = 12) -> begin
    x = nx ÷ 2 + nx * (col - 1)
    y = 4 + ny * (row - 1)
    return (x, y, (title, fontsize))
end
annotation = [
    annotate(1, 1, "ftrue"),
    annotate(2, 1, "fcoil"),
    annotate(3, 1, "funiform"),
    annotate(2, 2, "fpc_iterative"),
    annotate(3, 2, "fpc"),
]
fzero = zeros(eltype(ftrue), size(ftrue))
fmaps = [ftrue;;; fcoil;;; funiform;;; fzero;;; fpc_iterative;;; fpc]
jim(fmaps; clim, annotation, ncol = 3)

# Display the RMSE of each field map.
[
    "Method"       "RMSE";
    :fcoil         frmse(fcoil);
    :funiform      frmse(funiform);
    :fpc_iterative frmse(fpc_iterative);
    :fpc           frmse(fpc);
]

# Compute the root mean square difference
# between the iterative B0 mapping approach
# that used the phase contrast-based coil combination method
# and the conventional phase contrast B0 map.
println("RMSD = ", frmsd(fpc_iterative, fpc))


#=
## Summary

In this example,
the iterative B0-mapping method
that used the phase contrast-based coil combination method
performed just as well
(in terms of RMSE)
as the approach
that used coil-sensitivity information
for coil combination.
The iterative approach
that did coil combination
assuming uniform coil sensitivities
did slightly worse,
and the non-iterative phase contrast approach
had the worst RMSE.
However,
in this example,
all the approaches performed fairly well;
perhaps a different data set
would result in more dramatic differences.
=#
