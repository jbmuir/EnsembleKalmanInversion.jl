module EnsembleKalmanInversion

    using LowRankApprox: pheigfact, PartialHermitianEigen
    using Random: randperm
    import LinearAlgebra
    using LinearAlgebra: UniformScaling, I, Diagonal
    using Statistics: mean
    using Distributed:  pmap
    using EmpiricalCovarianceOperators
    
    export eki
    export heki

    include("utilities.jl")
    include("basiceki.jl")
    include("hierarchicaleki.jl")
 

end