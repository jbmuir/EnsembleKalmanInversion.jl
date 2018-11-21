module EKI

using LowRankApprox: AbstractLinearOperator, LinearOperator, pheigfact, PartialHermitianEigen
import Base: convert, size, transpose
using Compat

if VERSION < v"0.7"
    import Base: ishermitian, issymmetric
else
    using Random: randperm
    using Nullables
    import LinearAlgebra
    using LinearAlgebra: UniformScaling, I, Diagonal
    using Statistics: mean
    using Distributed: addprocs, pmap
end

export eki
export heki

include("covarianceoperator.jl")
include("ensemblekalmaninversion.jl")
include("hierarchicaleki.jl")
 

end