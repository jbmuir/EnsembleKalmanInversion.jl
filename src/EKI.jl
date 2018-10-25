module EKI

using LowRankApprox: AbstractLinearOperator, LinearOperator, pheigfact, PartialHermitianEigen
import Base: convert, size, transpose, ishermitian, issymmetric
using Compat
if VERSION >= v"0.7"
    using Random: randperm
    using Nullables
end
export eki
export heki

include("covarianceoperator.jl")
include("ensemblekalmaninversion.jl")
include("hierarchicaleki.jl")
 

end