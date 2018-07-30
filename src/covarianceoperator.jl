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

Base.convert(::Type{LinearOperator}, A::CovarianceOperator) = convert(LinearOperator{eltype(A)}, A)
Base.convert(::Type{LinearOperator{T}}, A::CovarianceOperator{T}) where T = LinearOperator{T}(A.p, A.p, A.mul!, A.mul!, A._tmp)
Compat.adjoint(A::CovarianceOperator) = A
Base.ishermitian(A::CovarianceOperator) = true
Base.issymmetric(A::CovarianceOperator) = isreal(A)
Base.size(A::CovarianceOperator) = (A.p, A.p)
Base.size(A::CovarianceOperator, dim::Integer) = (dim == 1 || dim == 2) ? A.p : 1
