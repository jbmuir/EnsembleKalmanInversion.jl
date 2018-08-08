function setγ(γ::R, ρ::R, y::Array{R,1}, wb::Array{R,1}, cww::Array{R,2}, Γ::Array{R,2}) where {R<:Real}
    while true
        rhs = ρ*sqrt((y-wb)'*(Γ\(y-wb)))
        tmp = (cww+γ.*Γ)\(y-wb)
        lhs = γ*sqrt(tmp'*(Γ*tmp))
        if lhs < rhs
            γ *= 2
        elseif lhs > 2*rhs
            γ /= 2
        else
            break
        end
    end
    γ
end


function eki(y::Array{R,1}, 
             σ::R, 
             η::R,
             J::Integer, 
             N::Integer, 
             prior::Function, 
             gmap::Function; 
             ρ::R = convert(R,0.5), 
             ζ::R = convert(R,0.5), 
             γ0::R = convert(R,1e0), 
             parallel::Bool = false, 
             verbosity=0, 
             lowmem=false) where R<:Real
    if lowmem
        eki_lowmem(y, σ, η, J, N, prior, gmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity)
    else
        ydim = length(y)
        Γ = σ^2.*eye(R,ydim)
        eki(y, Γ, η, J, N, prior, gmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity)
    end
end

function eki(y::Array{R,1}, 
             Γ::Array{R,2}, 
             η::R,
             J::Integer, 
             N::Integer, 
             prior::Function, 
             gmap::Function; 
             ρ::R = convert(R,0.5), 
             ζ::R = convert(R,2.0), 
             γ0::R = convert(R,1.0), 
             parallel::Bool = false,
             verbosity=0) where {R<:Real}
    #Initialization
    H = MvNormal(Γ)
    udim = length(prior())
    ydim = length(y)
    cuw = zeros(udim, ydim)
    cww = zeros(ydim, ydim)
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
        convg = sqrt((y-wb)'*(Γ\(y-wb)))
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)")
        end
        if convg <= ζ*η
            return mean(uj)
        end
        convg = sqrt((y-wb)'*(Γ\(y-wb)))
        #Analysis
        cuw[:] = cov(hcat(uj...)', hcat(wj...)')
        cww[:] = cov(hcat(wj...)', hcat(wj...)')
        γ = setγ(γ0, ρ, y, wb, cww, Γ)
        if verbosity > 0
            println("Iteration # $i. γ = $γ") 
        end
        for j = 1:J
            yj[j] = y + rand(H)
            uj[j] = uj[j] + cuw*((cww+γ.*Γ)\(yj[j]-wj[j]))
        end
    end
    mean(uj)
end
