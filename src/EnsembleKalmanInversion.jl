module EnsembleKalmanInversion

    using LowRankApprox: AbstractLinearOperator, LinearOperator, pheigfact, PartialHermitianEigen
    import Base: convert, size, transpose
    using Random: randperm
    using Nullables
    import LinearAlgebra
    using LinearAlgebra: UniformScaling, I, Diagonal
    using Statistics: mean
    using Distributed:  pmap

    export eki
    export heki

    include("covarianceoperator.jl")
    include("utilities.jl")
    include("basiceki.jl")
    include("hierarchicaleki.jl")
 

end