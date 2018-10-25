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
    if VERSION < v"0.7"
        Xm = mean(X,1)
    else
        Xm = mean(X, dims=1)
    end
    ml! = (y, _, x) -> covmul!(y, X.-Xm, n, x)
    CovarianceOperator{T}(p, ml!, nothing)
end

Base.convert(::Type{LinearOperator}, A::CovarianceOperator) = convert(LinearOperator{eltype(A)}, A)
Base.convert(::Type{LinearOperator{T}}, A::CovarianceOperator{T}) where T = LinearOperator{T}(A.p, A.p, A.mul!, A.mul!, A._tmp)
Base.transpose(A::CovarianceOperator) = A
if VERSION <= v"0.7"
    @compat adjoint(A::CovarianceOperator) = A
    Base.ishermitian(A::CovarianceOperator) = true
    Base.issymmetric(A::CovarianceOperator) = isreal(A)
else
    LinearAlgebra.adjoint(A::CovarianceOperator) = A
    LinearAlgebra.ishermitian(A::CovarianceOperator) = true
    LinearAlgebra.issymmetric(A::CovarianceOperator) = isreal(A)
end

Base.size(A::CovarianceOperator) = (A.p, A.p)
Base.size(A::CovarianceOperator, dim::Integer) = (dim == 1 || dim == 2) ? A.p : 1

function crosscovmul!(y, Xc, Yc, n, x)
    y = Xc'*(Yc*x)./(n-1)
end

mutable struct CrossCovarianceOperator{T<:Real} <: AbstractLinearOperator{T}
    p::Int
    q::Int
    mul!::Function
    mulc!::Function
    _tmp::Nullable{Array{T}}
end

function CrossCovarianceOperator(X, Y)
    if (size(X) == size(Y)) && isapprox(X, Y)
        return CovarianceOperator(X)
    end
    Tx = eltype(X)
    Ty = eltype(Y)
    @assert Tx == Ty
    nx, p = size(X)
    ny, q = size(Y)
    @assert nx == ny
    if VERSION < v"0.7"
        Xm = mean(X,1)
        Ym = mean(Y,1)
    else
        Xm = mean(X,dims=1)
        Ym = mean(Y,dims=1)
    end
    ml! = (y, _, x) -> crosscovmul!(y, X.-Xm, Y.-Ym, nx, x)
    mulc! = (y, _, x) -> crosscovmul!(y, Y.-Ym, X.-Xm, nx, x)
    CrossCovarianceOperator{Tx}(p, q, ml!, mulc!, nothing)
end

Base.convert(::Type{LinearOperator}, A::CrossCovarianceOperator) = convert(LinearOperator{eltype(A)}, A)
Base.convert(::Type{LinearOperator{T}}, A::CrossCovarianceOperator{T}) where T = LinearOperator{T}(A.p, A.q, A.mul!, A.mulc!, A._tmp)
Base.transpose(A::CrossCovarianceOperator{T}) where {T} = CrossCovarianceOperator{T}(A.q, A.p, A.mulc!, A.mul!, nothing)
if VERSION <= v"0.7"
    @compat adjoint(A::CrossCovarianceOperator{T}) where {T} = CrossCovarianceOperator{T}(A.q, A.p, A.mulc!, A.mul!, nothing)
    Base.ishermitian(A::CrossCovarianceOperator) = false
    Base.issymmetric(A::CrossCovarianceOperator) = false
else
    LinearAlgebra.adjoint(A::CrossCovarianceOperator{T}) where {T} = CrossCovarianceOperator{T}(A.q, A.p, A.mulc!, A.mul!, nothing)
    LinearAlgebra.ishermitian(A::CrossCovarianceOperator) = false
    LinearAlgebra.issymmetric(A::CrossCovarianceOperator) = false
end

Base.size(A::CrossCovarianceOperator) = (A.p, A.q)


#this is a ``read only'' shortcircuiting way to match the dim - see julia shortcircuiting rules to understand
Base.size(A::CrossCovarianceOperator, dim::Integer) = (dim == 1 && return A.p) || (dim == 2) ? A.q : 1
