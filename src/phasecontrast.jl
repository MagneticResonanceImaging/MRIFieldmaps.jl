# phasecontrast.jl

"""
    pc = phasecontrast(ydata)
    fhat = phasecontrast(ydata, echotime)

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
