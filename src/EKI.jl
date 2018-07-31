module EKI

using Distributions: MvNormal
using LowRankApprox: AbstractLinearOperator, LinearOperator, pheigfact
using WoodburyMatrices
import Base: convert, size, transpose, ishermitian, issymmetric
import Compat: adjoint

export eki
export heki

include("covarianceoperator.jl")
include("eki_lowmem.jl")
include("ensemblekalmaninversion.jl")
include("hierarchicaleki.jl")


end