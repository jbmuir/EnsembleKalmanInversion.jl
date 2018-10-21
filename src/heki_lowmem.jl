function heki_lowmem(y::Array{R,1}, 
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
                    verbosity=0) where {R<:Real}
    #Initialization
    Γ = σ^2 * I #this should use the efficient uniform scaling operator - can use both left & right division also
    T = σ^(-2) * I # Precision Matrix 
    uj = [prior() for j = 1:J]
    yj = [y for j = 1:J]
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
            wj = pmap(gmap, uj)
        else
            wj = map(gmap, uj)
        end
        wb = mean(wj)
        #Discrepancy principle
        convg = sqrt((y-wb)'*T*(y-wb))
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)")
        end
        if convg <= ζ*η
            return mean(uj)
        end
        #Analysis
        Cuw = CrossCovarianceOperator(hcat(uj...)', hcat(wj...)')
        Cwwf = pheigfact(CovarianceOperator(hcat(wj...)')) #this is a low-rank approx to Cww
        #set regularization
        γ = setγ_lowmem(γ0, ρ, T, Cwwf, y, wb)
        if verbosity > 0
            println("Iteration # $i. γ = $γ") 
        end
        for j = 1:J
            yj[j] = y.+σ.*randn(R, length(y))
            uj[j] = uj[j].+Cuw*(wbinv(T/γ, 
                                      Cwwf[:vectors], 
                                      diagm(convert(R,1.0)./Cwwf[:values]), 
                                      (yj[j]-wj[j])))
        γ0 = γ/2 # after first iteration, cannot reduce gamma by more than a factor of 2 each iteration...
        end
    end
    mean(uj)
end

