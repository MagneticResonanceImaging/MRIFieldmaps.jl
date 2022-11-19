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

    """
        Echotime{T} = Union{AbstractVector{<:T}, NTuple{N,<:T} where N}
    The echo times can be a vector or a Tuple.
    """
    Echotime{T} = Union{AbstractVector{<:T}, NTuple{N,<:T} where N}

    include("spdiff1.jl")
    include("spdiff2.jl")
    include("spdiff.jl")
    include("b0model.jl")
    include("b0init.jl")
    include("b0scale.jl")
    include("combine.jl")
    include("b0map.jl")

end # module
