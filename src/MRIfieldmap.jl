"""
    MRIfieldmap
Module `MRIfieldmap` exports methods for fieldmap estimation.
"""
module MRIfieldmap

    """
        RealU
    A data type that is just `Number`
    but is to be thought of as `Union{Real, Unitful.Length}`
    without needing a dependence on the `Unitful` package.
    """
    const RealU = Number # Union{Real, Unitful.Length}

    include("spdiff1.jl")
    include("spdiff2.jl")
    include("spdiff.jl")

end # module
