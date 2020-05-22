function eki(y::Union{Array{R,1}, Array{Array{R,1},1}},
             σ::R, 
             η::R,
             J::Integer, 
             N::Integer, 
             prior::Function, 
             gmap::Function; 
             ρ::R = convert(R,0.5), 
             ζ::R = convert(R,2.0), 
             γ0::R = convert(R,1e0), 
             parallel::Bool = false,
             verbosity::Int=0,
             rerandomize::Bool = false,
             rerandom_fun::Function = prior,
             rerandom_coeff::R = convert(R,0.25), 
             batched::Bool = false,
             batches::Int = 1) where R<:Real
    if batched
        @assert typeof(y) == Array{Array{R,1}, 1} "Must supply a list of data subsets for batched EKI, current type is $(typeof(y))"
        eki_batched(y, σ, η, J, N, prior, gmap, ρ, ζ, γ0, parallel, verbosity, rerandomize, rerandom_fun, rerandom_coeff, batches)
    else
        @assert typeof(y) == Array{R,1} "Must supply array of flat data for unbatched EKI, current type is $(typeof(y))"
        eki_nobatch(y, σ, η, J, N, prior, gmap, ρ, ζ, γ0, parallel, verbosity, rerandomize, rerandom_fun, rerandom_coeff)
    end
end


function ensemble_update!(u::Array{Array{R,1},1}, 
                          w::Array{Array{R,1},1}, 
                          wm::Array{R,1}, 
                          y::Array{R,1}, 
                          γ0::R,
                          σ::R, 
                          ρ::R, 
                          T::UniformScaling{R},
                          verbosity::Int) where R<:Real
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
            u[j] = u[j].+Cuw*(wbinv(T/γ, 
            Cwwf[:vectors], 
            Diagonal(convert(R,1.0)./Cwwf[:values]), 
            (yj-w[j])))
        end
end

function eki_nobatch(y::Array{R,1}, 
                    σ::R, 
                    η::R,
                    J::Integer, 
                    N::Integer, 
                    prior::Function, 
                    gmap::Function, 
                    ρ::R, 
                    ζ::R, 
                    γ0::R, 
                    parallel::Bool,
                    verbosity::Int,
                    rerandomize::Bool,
                    rerandom_fun::Function,
                    rerandom_coeff::R) where {R<:Real}
    #Initialization
    Γ = σ^2 * I #this should use the efficient uniform scaling operator - can use both left & right division also
    T = σ^(-2) * I # Precision Matrix 
    if parallel
        u = pmap(x->prior(), 1:J)
    else
        u = map(x->prior(), 1:J)
    end
    # u = [prior() for j = 1:J]
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
        #Gum = gmap(mean(u))
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
        if rerandomize
            um = mean(u)
            if parallel
                u = pmap(x-> um .+ rerandom_coeff .* rerandom_fun(), 1:J)
            else
                u = map(x-> um .+ rerandom_coeff .* rerandom_fun(), 1:J)
            end
        end
    end
    mean(u)
end


function eki_batched(y::Array{Array{R,1},1}, 
                    σ::R, 
                    η::R,
                    J::Integer, 
                    N::Integer, 
                    prior::Function, 
                    gmap::Function, 
                    ρ::R, 
                    ζ::R, 
                    γ0::R, 
                    parallel::Bool,
                    verbosity::Int,
                    rerandomize::Bool,
                    rerandom_fun::Function,
                    rerandom_coeff::R,
                    batches::Int) where {R<:Real}
    #Initialization
    Γ = σ^2 * I #this should use the efficient uniform scaling operator - can use both left & right division also
    T = σ^(-2) * I # Precision Matrix 
    if parallel
        u = pmap(x->prior(), 1:J)
    else
        u = map(x->prior(), 1:J)
    end
    #u = [prior() for j = 1:J]
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
        parts = kfoldperm(length(y), batches) #randomize batch groupings each iterations
        for (i, p) in enumerate(parts)
            if parallel
                w = pmap(x->vcat([gmap(x,b) for b in p]...), u)
            else
                w = map(x->vcat([gmap(x,b) for b in p]...), u)
            end
            wm = mean(w)
            ensemble_update!(u,w,wm,vcat(y[p]...),γ0,σ,ρ,T,verbosity)
            if rerandomize
                um = mean(u)
                if parallel
                    u = pmap(x-> um .+ rerandom_coeff .* rerandom_fun(), 1:J)
                else
                    u = map(x-> um .+ rerandom_coeff .* rerandom_fun(), 1:J)
                end
            end
        end
    end
    mean(u)
end

