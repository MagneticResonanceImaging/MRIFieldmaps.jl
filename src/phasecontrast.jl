# phasecontrast.jl

"""
    pc = phasecontrast(ydata)
    fhat = phasecontrast(ydata, echotime)

Compute the phase contrast between two multicoil data sets.
If `echotime` is provided,
return a field map
by converting from radians to Hz
(if `echotime` is in seconds, as is typical).

# In
- `ydata (dims..., nc, ne)` `ne` sets of complex images for `nc ≥ 1` coils
- `echotime::Echotime (ne = 2)` echo time offsets

# Out
- `pc (dims...)` phase contrast (or field map, if `echotime` is provided):
  `∠(sum_c ydata[c,1]' * ydata[c,2])`

See equation [13] in
M A Bernstein et al.,
"Reconstructions of Phase Contrast, Phased Array Multicoil Data", MRM 1994.
https://doi.org/10.1002/mrm.1910320308
"""
function phasecontrast(ydata::Array{<:Complex, D}) where {D}

    D < 4 && @warn("D = $D < 4 is appropriate only for 1D MRI")

    (nc, ne) = size(ydata)[end-1:end]

    ne > 2 && @warn("only first two echoes used")

    # The following is a dimensionality-agnostic way of writing
    # y1 = [ydata[:,:,:,c,1] for c = 1:nc]
    # y2 = [ydata[:,:,:,c,2] for c = 1:nc]
    # (where the above assumes 3D data)
    (y1, y2) = ntuple(2) do e
        ye = selectdim(ydata, D, e)
        [selectdim(ye, D - 1, c) for c = 1:nc]
    end

    # Eqn. [13] in the cited paper
    pc = mapreduce(+, y1, y2) do yc1, yc2
        conj.(yc1) .* yc2
    end

    return angle.(pc)

end


function phasecontrast(ydata::Array{<:Complex}, echotime::Echotime)

    ne = length(echotime)

    size(ydata)[end] == ne || throw("bad ydata size")

    pc = phasecontrast(ydata)

    ΔTE = echotime[begin+1] - echotime[begin]
    fhat = pc ./ (2π .* ΔTE)

    return fhat

end
