#---------------------------------------------------------
# # [Phase contrast B0 mapping](@id 03-phasecontrast)
#---------------------------------------------------------

#=
This page illustrates regularized B0 3D field map estimation
from multi-echo, multi-coil MRI images
using the Julia package
[`MRIFieldmaps`](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).
In this case,
we assume we do not have access to coil sensitivity maps.
Instead of using those maps for coil combination,
we will use a phase contrast-based approach.

This page was generated from a single Julia file:
[03-phasecontrast.jl](@__REPO_ROOT_URL__/03-phasecontrast.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`03-phasecontrast.ipynb`](@__NBVIEWER_ROOT_URL__/03-phasecontrast.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`03-phasecontrast.ipynb`](@__BINDER_ROOT_URL__/03-phasecontrast.ipynb).


# ### Setup

# Packages needed here.

using MRIFieldmaps: b0map
using MIRTjim: jim, prompt; jim(:prompt, true)
using Random: seed!


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
### Overview

When estimating B0 field maps
from multi-coil data,
the images first must be coil-combined
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
about the coil sensitivies
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
\Psi(\mathbf{\omega}) = \Phi(\mathrm{\omega}) + \frac{\beta}{2} \|\mathbf{C} \mathbf{\omega}\|_2^2,
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
and $\cdot^{*}$ denotes complex conjugate.

NOTE: It looks like `b0map` multiplies $r_{mnj}$ by $sos_j$.
TODO: Maybe update $r_{mnj}$ above to include multiplication by $sos_j$?

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
   z_{nj} = \frac{1}{v_j} \sum_{c = 1}^{N_{\mathrm{c}}} s_{cj}^{*} y_{cnj},
   ```
   where $y_{cnj}$ is the image data
   for coil $c$ of echo $n$ at voxel $j$.
   TODO: We then end up multiplying by $v_j$.
1. *Taking coil sensitivities to be uniformly equal to 1:*
   In this case,
   $s_{cj} = 1 \, \forall c, j$,
   so $v_j = N_{\mathrm{c}}$
   and
   ```math
   z_{nj} = \frac{1}{N_{\mathrm{c}}} \sum_{c = 1}^{N_{\mathrm{c}}} y_{cnj}.
   ```
   TODO: We then end up multiplying by $v_j$.
   TODO: Is β even scale independent?? I don't see where in the code that is the case.
1. *Phase contrast-based approach:*
   In this case,
   we first coil combine the first echo image
   by taking the square root sum-of-squares
   across coils:
   ```math
   v_j = \sqrt{\sum_{c = 1}^{N_{\mathrm{c}}} |y_{c1j}|^2},
   ```
   where $y_{c1j}$ is the image data
   for coil $c$ of the first echo at voxel $j$.
   Then, for each echo,
   we coil combine in the following way:
   ```math
   z_{nj} = \frac{1}{v_j} \sum_{c = 1}^{N_{\mathrm{c}}} y_{c1j}^{*} y_{cnj}.
   ```
   TODO: We then end up multiplying by $\frac{v_j}{\mathrm{max}_{j} v_j^2}$.
   TODO: Does it make more sense to divide by $|y_{c1j}|$ when computing $z_{nj}$?
   (Rather than dividing my $v_j$ and then multiplying by the above constant.)

This example shows a simulated experiment
where a B0 field map is estimated
without knowledge
of the coil sensitivies.
This example examines B0 maps estimated
when using both coil combination approaches
and compares them to the case
where coil sensitivity information is known.
This example also compares
the above regularized, iteratively estimated B0 maps
to one obtained
using a phase contrast B0 mapping approach.
=#

# ## Get simulated data

# ### Sensitivity maps

# Create a function for generating Gaussian-shaped, pure-real sensitivity maps.
function sensitivity_map(nx, ny, cx, cy, σ)

    s = [exp(-((x - cx)^2 + (y - cy)^2) / σ^2) for x = 1:nx, y = 1:ny]
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
obj = [(x - nx÷2)^2 + (y - ny÷2)^2 < radius^2 for x = 1:nx, y = 1:ny]
mask = obj .> 0 # Mask indicating object support
jim(obj; title = "Object")

# ### True field map

# Create a linearly (spatially) varying field map.
ftrue = repeat(range(-50, 50, nx), 1, ny) # Hz
ftrue .*= mask # Mask out background voxels for better visualization
jim(ftrue; title = "True field map")

# ### Data

# Specify parameters for the simulated multi-echo, multi-coil data.
TE = (4.0e-3, 6.3e-3) # s
T2 = 40e-3 # s
σ_noise = 0.005; # Noise standard deviation

# Make a function for creating the simulated data
# for a given echo time.
function make_data(TE)

    data = smap .* obj .* exp.(-TE / T2) .* cispi.(2 .* TE .* ftrue) # Noiseless
    data .+= complex.(σ_noise .* randn.(), σ_noise .* randn.()) # Add complex Gaussian noise
    return data

end;

# Create the simulated data.
seed!(0)
ydata = cat(make_data.(TE)...; dims = 4)
jim([ydata[:,:,:,1];;; ydata[:,:,:,2]]; title = "Simulated data")


# ## Estimate B0

# Now that we have the simulated data,
# we can estimate B0 from the data
# using the different approaches.

# First, create a function for computing the RMSE of the estimated B0 maps.
frmse = f -> round(sqrt(sum(abs2, (f - ftrue)[mask]) / count(mask)); digits = 1);

# ### Assume uniform coil sensitivities


