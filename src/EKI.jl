module EKI

using Distributions: MvNormal
using LowRankApprox: AbstractLinearOperator, LinearOperator, pheigfact, PartialHermitianEigen
import Base: convert, size, transpose, ishermitian, issymmetric
using Compat

export eki
export heki

include("covarianceoperator.jl")
include("eki_lowmem.jl")
include("ensemblekalmaninversion.jl")
include("hierarchicaleki.jl")
 

end