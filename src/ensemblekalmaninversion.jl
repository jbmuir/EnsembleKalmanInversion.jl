function setγ(γ, ρ, y, wb, cww, Γ)
    while true
        rhs = ρ*sqrt((y-wb)'*(Γ\(y-wb)))
        tmp = (cww+γ.*Γ)\(y-wb)
        lhs = γ*sqrt(tmp'*(Γ*tmp))
        if lhs < rhs
            γ *= 2
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
             γ0::R = convert(R,0.5), 
             parallel::Bool = false, 
             verbosity=0) where R<:Real
    ydim = length(y)
    Γ = σ^2.*eye(R,ydim)
    eki(y, Γ, η, J, N, prior, gmap; ρ = ρ, ζ = ζ, γ0 = γ0, parallel = parallel, verbosity=verbosity)
end

function eki(y::Array{R,1}, 
                Γ::Array{R,2}, 
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
            ub = mean(uj)
            return ub
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

function setγ_lowmem(γ::R, ρ::R, y::Array{R,1}, wb::Array{R,1}, cww::Array{R,2}, σ2::R) where {R<:Real}
    while true
        rhs = ρ*sqrt((y-wb)'*(y-wb)/σ2)
        tmp = (cww.+γ.*σ2.*eye(R, length(y)))\(y-wb)
        lhs = γ*sqrt(tmp'*(σ2.*tmp))
        if lhs < rhs
            γ *= 2
        else
            break
        end
    end
    γ
end

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
        println("Mean uj")
        println(mean(uj))
        #Discrepancy principle
        convg = sqrt((y-wb)'*(y-wb)/σ2)
        if verbosity > 0
            println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)")
        end
        if convg <= ζ*η
            ub = mean(uj)
            return ub
        end
        convg = sqrt((y-wb)'*(y-wb)/σ2)
        #Analysis
        cuw[:] = cov(hcat(uj...)', hcat(wj...)')
        cww[:] = cov(hcat(wj...)', hcat(wj...)')
        γ = γ0
        γc = 0
        #γ = setγ_lowmem(γ0, ρ, y, wb, cww, σ2)
        #actually just copy in the function to save memory...
        while true
            rhs = ρ*sqrt((y-wb)'*(y-wb)/σ2)
            for j = 1:ydim
                cww[j,j] += (γ-γc)*σ2
            end
            tmp = cww\(y-wb)
            lhs = γ*sqrt(tmp'*(σ2.*tmp))
            println(lhs," ", rhs)
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
        #cwwf = cholfact(cww.+γ.*σ2.*eye(R,ydim))
        cwwf = cholfact(cww) #now cww is already incorporating the regularization
        for j = 1:J
            yj[j] = y.+σ.*randn(R, ydim)
            uj[j] = uj[j].+cuw*(cwwf\(yj[j]-wj[j]))
        end
    end
    mean(uj)
end