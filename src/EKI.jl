module EKI

using Distributions

export eki, eki_lowmem

include("ensemblekalmaninversion.jl")
include("hierarchicaleki.jl")


end