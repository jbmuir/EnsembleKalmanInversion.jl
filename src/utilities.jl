function wbinv(Ai::UniformScaling{R}, B::Array{R,2}, Ci::Diagonal{R,Array{R,1}}, x::Array{R,1}) where {R<:Real}
    #woodbury inverse for A given by uniformscaling Ai = A inverse
    # Ci = C inverse
    # B = U, V (in our case UCV = B'CB is symmetric)
    Ai*x - (Ai*B)*(((Ci+B'*Ai*B)\B')*(Ai*x))
end

function wbinv(Ai::UniformScaling{R}, B, Ci, x) where {R<:Real}
    #woodbury inverse for A given by uniformscaling Ai = A inverse
    # Ci = C inverse
    # B = U, V (in our case UCV = B'CB is symmetric)
    Ai*x - (Ai*B)*(((Ci+B'*Ai*B)\B')*(Ai*x))
end


function kfoldperm(N,k)
    #function from Dan Getz  https://stackoverflow.com/questions/37989159/how-to-divide-my-data-into-distincts-mini-batches-randomly-julia
    #because my brain was shutoff on a monday afternoon
    n,r = divrem(N,k)
    b = collect(1:n:N+1)
    for i in 1:length(b)
        b[i] += i > r ? r : i-1  
    end
    p = randperm(N)
    return [p[r] for r in [b[i]:b[i+1]-1 for i=1:k]]
end


function setγ(γ::R, 
    ρ::R,
    T::UniformScaling{R}, 
    Cwwf::PartialHermitianEigen{R,R}, 
    y::Array{R,1}, 
    wb::Array{R,1}) where {R<:Real}
    while true
        rhs = ρ*sqrt((y-wb)'*T*(y-wb))
        tmp = wbinv(T/γ, Cwwf[:vectors], Diagonal(convert(R,1.0)./Cwwf[:values]), y-wb)
        lhs = γ*sqrt(tmp'*tmp/T.λ)
        if lhs < rhs
            γ *= 2
        else
            break
        end
    end
    γ
end