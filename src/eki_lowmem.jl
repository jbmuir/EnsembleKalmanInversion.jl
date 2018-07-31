#TODO: can we improve this further using low-rank approximations of cww?
#      rank of cww is at most J?

function wbinv(Ai::UniformScaling{R}, B, Ci, x) where {R<:Real}
    #woodbury inverse for A given by uniformscaling Ai = A inverse
    # Ci = C inverse
    # B = U, V (in our case UCV = B'CB is symmetric)
    Ai*x - Ai*B*(Ci+B'*Ai*B)*B'*Ai*x
end


function setγ_lowmem(γ::R, 
                     ρ::R,
                     T::UniformScaling{R}, 
                     Cwwf::PartialHermitianEigen{R,R}, 
                     y::Array{R,1}, 
                     wb::Array{R,1}) where {R<:Real}
    while true
        rhs = ρ*sqrt((y-wb)'*T*(y-wb))
        tmp = wbinv(T/γ, Cwwf[:vectors], diagm(convert(R,1.0)./Cwwf[:values]), y-wb)
        lhs = γ*sqrt(tmp'*tmp/T.λ)
        println(lhs, " ", rhs)
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
        end
    end
    mean(uj)
end

#OLD VERSION: DELETE ONCE OBSELETE
# function eki_lowmem(y::Array{R,1}, 
#                     σ::R, 
#                     η::R,
#                     J::Integer, 
#                     N::Integer, 
#                     prior::Function, 
#                     gmap::Function; 
#                     ρ::R = convert(R,0.5), 
#                     ζ::R = convert(R,0.5), 
#                     γ0::R = convert(R,1.0), 
#                     parallel::Bool = false,
#                     verbosity=0) where {R<:Real}
#     #Initialization
#     σ2 = σ^2
#     udim = length(prior())
#     ydim = length(y)
#     cuw = zeros(R,udim, ydim)
#     cww = zeros(R,ydim, ydim)
#     uj = [prior() for j = 1:J]
#     yj = [y for j = 1:J]
#     #Main Optimization loop
#     if verbosity > 0
#         println("Starting up to $N iterations with $J ensemble members")
#     end
#     for i = 1:N
#         #Prediction
#         if parallel
#             wj = pmap(gmap, uj)
#         else
#             wj = map(gmap, uj)
#         end
#         wb = mean(wj)
#         #Discrepancy principle
#         convg = sqrt((y-wb)'*(y-wb)/σ2)
#         if verbosity > 0
#             println("Iteration # $i. Discrepancy Check; Weighted Norm: $convg, Noise level: $(ζ*η)")
#         end
#         if convg <= ζ*η
#             return mean(uj)
#         end
#         convg = sqrt((y-wb)'*(y-wb)/σ2)
#         #Analysis
#         cuw[:] = cov(hcat(uj...)', hcat(wj...)')
#         cww[:] = cov(hcat(wj...)', hcat(wj...)')
#         γ = γ0
#         γc = 0
#         #set regularization
#         while true
#             rhs = ρ*sqrt((y-wb)'*(y-wb)/σ2)
#             for j = 1:ydim
#                 cww[j,j] += (γ-γc)*σ2
#             end
#             tmp = cww\(y-wb)
#             lhs = γ*sqrt(tmp'*(σ2.*tmp))
#             if lhs < rhs
#                 γc = γ
#                 γ *= 2
#             else
#                 break
#             end
#         end
#         if verbosity > 0
#             println("Iteration # $i. γ = $γ") 
#         end
#         cwwf = cholfact(cww) #now cww is already incorporating the regularization
#         for j = 1:J
#             yj[j] = y.+σ.*randn(R, ydim)
#             uj[j] = uj[j].+cuw*(cwwf\(yj[j]-wj[j]))
#         end
#     end
#     mean(uj)
# end
