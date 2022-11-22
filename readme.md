# MRIfieldmaps

https://github.com/MagneticResonanceImaging/MRIfieldmaps.jl


[![action status][action-img]][action-url]
[![codecov][codecov-img]][codecov-url]
[![license][license-img]][license-url]
[![docs-stable][docs-stable-img]][docs-stable-url]
[![docs-dev][docs-dev-img]][docs-dev-url]
[![code-style][code-blue-img]][code-blue-url]

This Julia language repo
provides methods
for regularized estimation of fieldmaps in MRI.

Currently there are methods for B0 fieldmap estimation
from a water-only image signal model,
or from a water-fat model
with 1 or more lipid peaks.


## Getting started

```julia
using Pkg
Pkg.add("MRIfieldmaps")
```


## Example

For examples,
see the
[documentation](https://jefffessler.github.io/MRIfieldmaps.jl/stable).

```julia
images = ... # complex images of size (nx, ny, ..., ncoil, nechotime)
images = ComplexF32.(imges) # 32-bit floats saves memory and thus time
echotime = [0, 2] * 1f-3 # echo times in seconds
b0fieldmap, _, _ = b0map(images, echotime) # regularized fieldmap in Hz
```

### Citations

The algorithm in function `b0map`
for B0 field map estimation is based on the paper:
C Y Lin, J A Fessler,
"Efficient Regularized Field Map Estimation in 3D MRI", IEEE TCI 2020
[http://doi.org/10.1109/TCI.2020.3031082]
[http://arxiv.org/abs/2005.08661]
Please cite this paper if you use this method.

The internal algorithm details are a bit different
(and faster)
because here we perform coil combination
before starting the iterations,
whereas
[the original Matlab code](https://github.com/ClaireYLin/regularized-field-map-estimation)
had loops over coils
within each iteration.


### Compatibility

Tested with Julia ≥ 1.7.


### Related packages

* https://github.com/MagneticResonanceImaging

* https://github.com/ClaireYLin/regularized-field-map-estimation

<!-- URLs -->
[action-img]: https://github.com/MagneticResonanceImaging/MRIfieldmaps.jl/workflows/CI/badge.svg
[action-url]: https://github.com/MagneticResonanceImaging/MRIfieldmaps.jl/actions
[build-img]: https://github.com/MagneticResonanceImaging/MRIfieldmaps.jl/workflows/CI/badge.svg?branch=main
[build-url]: https://github.com/MagneticResonanceImaging/MRIfieldmaps.jl/actions?query=workflow%3ACI+branch%3Amain
[code-blue-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[code-blue-url]: https://github.com/invenia/BlueStyle
[codecov-img]: https://codecov.io/github/MagneticResonanceImaging/MRIfieldmaps.jl/coverage.svg?branch=main
[codecov-url]: https://codecov.io/github/MagneticResonanceImaging/MRIfieldmaps.jl?branch=main
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://MagneticResonanceImaging.github.io/MRIfieldmaps.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://MagneticResonanceImaging.github.io/MRIfieldmaps.jl/dev
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE
