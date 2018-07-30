#TODO: can we improve this further using low-rank approximations of cww?
#      rank of cww is at most J?

#Notes: 
#CovarianceOperator is a custom hermitian operator that explicitly defines the multiplication operation
#so that you never need the whole covariance matrix in memory
#Why is there a tmp here? 
function covmul!(y, Xc, n, x)
    y = Xc'*(Xc*x)./(n-1)
end

mutable struct CovarianceOperator{T<:Real} <: AbstractLinearOperator{T}
    p::Int
    mul!::Function
    _tmp::Nullable{Array{T}}
end

function CovarianceOperator(X)
    T = eltype(X)
    n, p = size(X)
    Xm = mean(X,1)
    ml! = (y, _, x) -> covmul!(y, X.-Xm, n, x)
    CovarianceOperator{T}(p, ml!, nothing)
end

convert(::Type{LinearOperator}, A::CovarianceOperator) = convert(LinearOperator{eltype(A)}, A)
convert(::Type{LinearOperator{T}}, A::CovarianceOperator{T}) where T = LinearOperator{T}(A.p, A.p, A.mul!, A.mul!, A._tmp)
adjoint(A::CovarianceOperator) = A
ishermitian(::CovarianceOperator) = true
issymmetric(A::CovarianceOperator) = isreal(A)
size(A::CovarianceOperator) = (A.p, A.p)
size(A::CovarianceOperator, dim::Integer) = (dim == 1 || dim == 2) ? A.p : 1


function eki_lowmem(y::Array{R,1}, 
                    σ::R, 
                    η::R,
                    J::Integer, 
                    N::Integer, 
                    prior::Function, 
                    gmap::Function; 
                    ρ::R = convert(R,0.5), 
                    ζ::R = convert(R,0.5), 
                    γ0::R = convert(R,1.0), 
                    parallel::Bool = false,
                    verbosity=0) where {R<:Real}
    #Initialization
    σ2 = σ^2
    udim = length(prior())
    ydim = length(y)
    cuw = zeros(R,udim, ydim)
    cww = zeros(R,ydim, ydim)
    uj = [prior() for j = 1:J]
    yj = [y for j = 1:J]
    #Main Optimization loop
    if verbosity > 0
        println("Starting up to $N iterations with $J ensemble members")
    end
    for i = 1:N
        #Prediction
        if parallel
            wj = pmap(gmap, uj)
        else
            wj = map(gmap, uj)
        end
        wb = mean(wj)
        #Discrepancy principle
        convg = sqrt((y-wb)'*(y-wb)/σ2)
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)")
        end
        if convg <= ζ*η
            return mean(uj)
        end
        convg = sqrt((y-wb)'*(y-wb)/σ2)
        #Analysis
        cuw[:] = cov(hcat(uj...)', hcat(wj...)')
        cww[:] = cov(hcat(wj...)', hcat(wj...)')
        γ = γ0
        γc = 0
        #set regularization
        while true
            rhs = ρ*sqrt((y-wb)'*(y-wb)/σ2)
            for j = 1:ydim
                cww[j,j] += (γ-γc)*σ2
            end
            tmp = cww\(y-wb)
            lhs = γ*sqrt(tmp'*(σ2.*tmp))
            if lhs < rhs
                γc = γ
                γ *= 2
            else
                break
            end
        end
        if verbosity > 0
            println("Iteration # $i. γ = $γ") 
        end
        cwwf = cholfact(cww) #now cww is already incorporating the regularization
        for j = 1:J
            yj[j] = y.+σ.*randn(R, ydim)
            uj[j] = uj[j].+cuw*(cwwf\(yj[j]-wj[j]))
        end
    end
    mean(uj)
end