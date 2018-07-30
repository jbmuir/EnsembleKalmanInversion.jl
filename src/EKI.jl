module EKI

using Distributions: MvNormal
using LowRankApprox
import Base.convert, Base.size

export eki, eki_lowmem
export heki

include("ensemblekalmaninversion.jl")
include("eki_lowmem.jl")
include("hierarchicaleki.jl")


end