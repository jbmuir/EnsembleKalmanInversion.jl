function heki(y::Array{Array{R,1},1}, 
             σ::R, 
             η::R,
             J::Integer, 
             N::Integer, 
             priorτ::Function, 
             priorθ::Function,
             gmap::Function,
             tmap::Function; 
             ρ::R = convert(R,0.5), 
             ζ::R = convert(R,2.0), 
             γ0::R = convert(R,1e0), 
             batched::Bool = false,
             batches::Int = 1,
             parallel::Bool = false,
             verbosity::Int=0,
             rerandomize::Bool = false,
             rerandom_coeff::R = convert(R,0.25)) where R<:Real
    if batched
        heki_batched(y, σ, η, J, N, priorτ, priorθ, gmap, tmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity, rerandomize=rerandomize, rerandom_coeff=rerandom_coeff, batches=batches, batch_off=batch_off)
    else
        @assert length(y) == 1 "y must have only one element in non-batched form"
        heki_nobatch(y[1], σ, η, J, N, priorτ, priorθ, gmap, tmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity, rerandomize=rerandomize, rerandom_coeff=rerandom_coeff)
    end
end

function heki_ensemble_update!(τ::Array{Array{R,1},1}, 
                               θ::Array{Array{R,1},1},
                               w::Array{Array{R,1},1}, 
                               wm::Array{R,1}, 
                               y::Array{R,1}, 
                               γ0::R,
                               σ::R, 
                               ρ::R, 
                               T::UniformScaling{R},
                               verbosity::Int=0) where R<:Real
        #Analysis
        Cτw = CrossCovarianceOperator(hcat(τ...)', hcat(w...)')
        Cθw = CrossCovarianceOperator(hcat(θ...)', hcat(w...)')
        Cwwf = pheigfact(CovarianceOperator(hcat(w...)')) #this is a low-rank approx to Cww
        #set regularization
        γ = setγ(γ0, ρ, T, Cwwf, y, wm)
        if verbosity >= 2
            println("γ = $γ") 
        end
        for j = 1:size(τ)[1]
            yj = y.+σ.*randn(R, length(y))
            if VERSION < v"0.7"
                τ[j] = τ[j].+Cτw*(wbinv(T/γ, 
                                        Cwwf[:vectors], 
                                        diagm(convert(R,1.0)./Cwwf[:values]), 
                                        (yj-w[j])))
                θ[j] = θ[j].+Cθw*(wbinv(T/γ, 
                                        Cwwf[:vectors], 
                                        diagm(convert(R,1.0)./Cwwf[:values]), 
                                        (yj-w[j])))
            else
                τ[j] = τ[j].+Cτw*(wbinv(T/γ, 
                Cwwf[:vectors], 
                Diagonal(convert(R,1.0)./Cwwf[:values]), 
                (yj-w[j])))
                θ[j] = θ[j].+Cθw*(wbinv(T/γ, 
                Cwwf[:vectors], 
                Diagonal(convert(R,1.0)./Cwwf[:values]), 
                (yj-w[j])))
            end
        end
end



function heki_nobatch(y::Array{R,1}, 
                      σ::R, 
                      η::R,
                      J::Integer, 
                      N::Integer, 
                      priorτ::Function, 
                      priorθ::Function,
                      gmap::Function,
                      tmap::Function; 
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
    τ = [priorτ() for j = 1:J]
    θ = [priorθ() for j = 1:J]
    tmap1arg(x) = tmap(x[1], x[2])
    #Main Optimization loop
    if verbosity > 0
        println("Starting up to $N iterations with $J ensemble members")
    end
    for i = 1:N
        if verbosity > 0
            println("Iteration # $i. Starting Forward Map")
        end
        #Prediction
        #project hierarchical parameters to parameters of interest
        if parallel
            u = pmap(tmap1arg, zip(τ, θ))
        else
            u = map(tmap1arg, zip(τ, θ))
        end
        #predict data
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
            return (mean(τ), mean(θ))
        end
        heki_ensemble_update!(τ,θ,w,wm,y,γ0,σ,ρ,T,verbosity)
    end
    (mean(τ), mean(θ))
end


function heki_batched(y::Array{Array{R,1},1}, 
                    σ::R, 
                    η::R,
                    J::Integer, 
                    N::Integer, 
                    priorτ::Function, 
                    priorθ::Function,
                    gmap::Function,
                    tmap::Function; 
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
    τ = [priorτ() for j = 1:J]
    θ = [priorθ() for j = 1:J]
    tmap1arg(x) = tmap(x[1], x[2])
    ya = vcat(y...)
    #Main Optimization loop
    if verbosity > 0
        println("Starting up to $N iterations with $J ensemble members")
    end
    for i = 1:N
        if verbosity > 0
            println("Iteration # $i. Starting Forward Map") 
        end
        #project hierarchical parameters to parameters of interest
        if parallel
            u = pmap(tmap1arg, zip(τ, θ))
        else
            u = map(tmap1arg, zip(τ, θ))
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
            return (mean(τ), mean(θ))
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
            ensemble_update!(τ,θ,w,wm,vcat(y[p]...),γ0,σ,ρ,T,verbosity)
        end
    end
    (mean(τ), mean(θ))
end


