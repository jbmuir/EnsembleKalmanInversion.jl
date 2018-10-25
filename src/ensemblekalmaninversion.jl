#TODO: can we improve this further using low-rank approximations of cww?
#      rank of cww is at most J?

function eki(y::Array{Array{R,1},1}, 
             σ::R, 
             η::R,
             J::Integer, 
             N::Integer, 
             prior::Function, 
             gmap::Function; 
             ρ::R = convert(R,0.5), 
             ζ::R = convert(R,2.0), 
             γ0::R = convert(R,1e0), 
             batched::Bool = false,
             batches::Int = 1,
             batch_off::Int = 20,
             parallel::Bool = false,
             verbosity::Int=0,
             rerandomize::Bool = false,
             rerandom_coeff::R = convert(R,0.25)) where R<:Real
    if batched
        eki_batched(y, σ, η, J, N, prior, gmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity, rerandomize=rerandomize, rerandom_coeff=rerandom_coeff, batches=batches, batch_off=batch_off)
    else
        @assert length(y) == 1 "y must have only one element in non-batched form"
        eki_nobatch(y[1], σ, η, J, N, prior, gmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity, rerandomize=rerandomize, rerandom_coeff=rerandom_coeff)
    end
end

function wbinv(Ai::UniformScaling{R}, B, Ci, x) where {R<:Real}
    #woodbury inverse for A given by uniformscaling Ai = A inverse
    # Ci = C inverse
    # B = U, V (in our case UCV = B'CB is symmetric)
    Ai*x - (Ai*B)*(((Ci+B'*Ai*B)\B')*Ai)*x
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
        if VERSION < v"0.7"
            tmp = wbinv(T/γ, Cwwf[:vectors], diagm(convert(R,1.0)./Cwwf[:values]), y-wb)
        else
            tmp = wbinv(T/γ, Cwwf[:vectors], Diagonal(convert(R,1.0)./Cwwf[:values]), y-wb)
        end
        lhs = γ*sqrt(tmp'*tmp/T.λ)
        if lhs < rhs
            γ *= 2
        else
            break
        end
    end
    γ
end

function ensemble_update!(u::Array{Array{R,1},1}, 
                          w::Array{Array{R,1},1}, 
                          wm::Array{R,1}, 
                          y::Array{R,1}, 
                          γ0::R,
                          σ::R, 
                          ρ::R, 
                          T::UniformScaling{R},
                          verbosity::Int=0) where R<:Real
        #Analysis
        Cuw = CrossCovarianceOperator(hcat(u...)', hcat(w...)')
        Cwwf = pheigfact(CovarianceOperator(hcat(w...)')) #this is a low-rank approx to Cww
        #set regularization
        γ = setγ(γ0, ρ, T, Cwwf, y, wm)
        if verbosity >= 2
            println("γ = $γ") 
        end
        for j = 1:size(u)[1]
            yj = y.+σ.*randn(R, length(y))
            if VERSION < v"0.7"
                u[j] = u[j].+Cuw*(wbinv(T/γ, 
                                        Cwwf[:vectors], 
                                        diagm(convert(R,1.0)./Cwwf[:values]), 
                                        (yj-w[j])))
            else
                u[j] = u[j].+Cuw*(wbinv(T/γ, 
                Cwwf[:vectors], 
                Diagonal(convert(R,1.0)./Cwwf[:values]), 
                (yj-w[j])))
            end
        end
end

function eki_nobatch(y::Array{R,1}, 
                    σ::R, 
                    η::R,
                    J::Integer, 
                    N::Integer, 
                    prior::Function, 
                    gmap::Function; 
                    ρ::R = convert(R,0.5), 
                    ζ::R = convert(R,2.0), 
                    γ0::R = convert(R,1.0), 
                    parallel::Bool = false,
                    verbosity::Int=0,
                    rerandomize::Bool = false,
                    rerandom_coeff::R = convert(R,0.25)) where {R<:Real}
    #Initialization
    Γ = σ^2 * I #this should use the efficient uniform scaling operator - can use both left & right division also
    T = σ^(-2) * I # Precision Matrix 
    u = [prior() for j = 1:J]
    #Main Optimization loop
    if verbosity > 0
        println("Starting up to $N iterations with $J ensemble members")
    end
    for i = 1:N
        if verbosity > 0
            println("Iteration # $i. Starting Forward Map")
        end
        #Prediction
        if parallel
            w = pmap(gmap, u)
        else
            w = map(gmap, u)
        end
        wm = mean(w)
        #Discrepancy principle
        convg = sqrt((y-wm)'*T*(y-wm))
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)") 
        end
        if convg <= ζ*η
            return mean(u)
        end
        ensemble_update!(u,w,wm,y,γ0,σ,ρ,T,verbosity)
    end
    mean(u)
end


function eki_batched(y::Array{Array{R,1},1}, 
                    σ::R, 
                    η::R,
                    J::Integer, 
                    N::Integer, 
                    prior::Function, 
                    gmap::Function; 
                    ρ::R = convert(R,0.5), 
                    ζ::R = convert(R,2.0), 
                    γ0::R = convert(R,1.0), 
                    parallel::Bool = false,
                    verbosity::Int=0,
                    rerandomize::Bool = false,
                    rerandom_coeff::R = convert(R,0.25),
                    batches::Int=1,
                    batch_off::Int=20) where {R<:Real}
    #Initialization
    Γ = σ^2 * I #this should use the efficient uniform scaling operator - can use both left & right division also
    T = σ^(-2) * I # Precision Matrix 
    u = [prior() for j = 1:J]
    ya = vcat(y...)
    #Main Optimization loop
    if verbosity > 0
        println("Starting up to $N iterations with $J ensemble members")
    end
    for i = 1:N
        if verbosity > 0
            println("Iteration # $i. Starting Forward Map") 
        end
        #Prediction
        if parallel
            w = pmap(x->vcat([gmap(x,b) for b in 1:length(y)]...), u)
        else
            w = map(x->vcat([gmap(x,b) for b in 1:length(y)]...), u)
        end
        #Discrepancy principle
        wm = mean(w)
        convg = sqrt((ya-wm)'*T*(ya-wm))
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)") 
        end
        if convg <= ζ*η
            return mean(u)
        end
        #set up batch ordering
        if i < batch_off
            parts = kfoldperm(length(y), batches) #randomize batch groupings each iterations
        else
            parts = kfoldperm(length(y), 1) #turn off minibatching after you have approached the minimum
        end
        for (i, p) in enumerate(parts)
            if parallel
                w = pmap(x->vcat([gmap(x,b) for b in p]...), u)
            else
                w = map(x->vcat([gmap(x,b) for b in p]...), u)
            end
            wm = mean(w)
            ensemble_update!(u,w,wm,vcat(y[p]...),γ0,σ,ρ,T,verbosity)
        end
    end
    mean(u)
end

