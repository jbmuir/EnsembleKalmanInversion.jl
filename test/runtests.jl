#!/usr/bin/env julia

module EnsembleKalmanInversionTests

    #Start Test Script
    using EnsembleKalmanInversion
    using Test
    using Random
    using Statistics
    
    # Run tests
    
    @testset "EnsembleKalmanInversion Tests" begin

        @testset "EnsembleKalmanInversion Tests" begin
            @testset "Linear Test" begin
            #This should certainly be a set of "convergence" tests
                G = randn(50,100)
                x = randn(100)
                σ = 0.1
                yt = G*x
                y = yt .+ σ * randn(50)
                η = sqrt((y-yt)'*(y-yt)/σ^2)
                prior() = randn(100)
                gmap(q) = G*q 
                J = 100
                N = 100
                xe = eki(y, σ, η, J, N, prior, gmap; verbosity=1, ρ=0.5, ζ=2.0) 
                @test sqrt((yt-G*xe)'*((yt-G*xe))/σ^2) <= 2.0*η 
            end    

            @testset "Linear Batched Test" begin
            #This should certainly be a set of "convergence" tests
                G1 = randn(10,100)
                G2 = randn(15,100)
                G3 = randn(13,100)
                G4 = randn(11,100)

                x = randn(100)
                σ = 0.1
                yt = [G1*x,G2*x,G3*x,G4*x]
                y = [G1*x.+σ*randn(10),G2*x.+σ*randn(15),G3*x.+σ*randn(13),G4*x.+σ*randn(11)]
                η = sqrt((y-yt)'*(y-yt)/σ^2)
                prior() = randn(100)
                function gmap(q,b)
                    if b <=1
                        return G1*q
                    elseif b <= 2
                        return G2*q
                    elseif b <= 3
                        return G3*q
                    else
                        return G4*q
                    end
                end
                J = 100
                N = 100
                xe = eki(y, σ, η, J, N, prior, gmap; verbosity=1, ρ=0.5, ζ=2.0, batched=true, batches=3) 
                ye = [G1*xe,G2*xe,G3*xe,G4*xe]
                @test sqrt((vcat(yt...)-vcat(ye...))'*((vcat(y...)-vcat(ye...)))/σ^2) <= 2.0*η 
            end    
        end
    end
end