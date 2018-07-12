module EKI

using Distributions

export eki, eki_lowmem

include("ensemblekalmaninversion.jl")
include("eki_lowmem.jl")
include("hierarchicaleki.jl")


end