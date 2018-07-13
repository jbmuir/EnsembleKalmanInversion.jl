function heki(y::Array{R,1}, 
              Γ::Array{R,2}, 
              η::R,
              J::Integer, 
              N::Integer, 
              priorτ::Function,
              priorθ::Function, 
              gmap::Function,
              tmap::Function; 
              ρ::R = convert(R,0.5), 
              ζ::R = convert(R,0.5), 
              γ0::R = convert(R,1.0), 
              parallel::Bool = false,
              verbosity=0) where {R<:Real}
    #note that depending on the algorithm for gmaph, this function can represent either
    #the centered (tmap(τ,θ) = τ with θ given to the function but dropped) or the non-centered version
    #(u= tmap(τ,θ)). In the centered case, just identify τ=u
    H = MvNormal(Γ)
    τdim = length(priorτ())
    θdim = length(priorθ())
    ydim = length(y)
    cτw = zeros(τdim, ydim)
    cθw = zeros(θdim, ydim)
    cww = zeros(ydim, ydim)
    τj = [priorτ() for j = 1:J]
    θj = [priorθ() for j = 1:J]
    tmap1arg(x) = tmap(x[1], x[2])
    yj = [y for j = 1:J]
    #Main Optimization loop
    if verbosity > 0
        println("Starting up to $N iterations with $J ensemble members")
    end
    for i = 1:N
        #project hierarchical parameters to parameters of interest
        if parallel
            uj = pmap(tmap1arg, zip(τj, θj))
        else
            uj = map(tmap1arg, zip(τj, θj))
        end
        #Prediction
        if parallel
            wj = pmap(gmap, uj)
        else
            wj = map(gmap, uj)
        end
        wb = mean(wj)
        #Discrepancy principle
        convg = sqrt((y-wb)'*(Γ\(y-wb)))
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)")
        end
        if convg <= ζ*η
            return (mean(uj), mean(τj), mean(θj))
        end
        convg = sqrt((y-wb)'*(Γ\(y-wb)))
        #Analysis
        cτw[:] = cov(hcat(τj...)', hcat(wj...)')
        cθw[:] = cov(hcat(θj...)', hcat(wj...)')
        cww[:] = cov(hcat(wj...)', hcat(wj...)')
        γ = setγ(γ0, ρ, y, wb, cww, Γ)
        if verbosity > 0
            println("Iteration # $i. γ = $γ") 
        end
        for j = 1:J
            yj[j] = y + rand(H)
            τj[j] = τj[j] + cτw*((cww+γ.*Γ)\(yj[j]-wj[j]))
            θj[j] = θj[j] + cθw*((cww+γ.*Γ)\(yj[j]-wj[j]))
        end
    end
    (mean(uj), mean(τj), mean(θj))
end
