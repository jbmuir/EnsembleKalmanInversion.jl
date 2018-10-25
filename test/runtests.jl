#!/usr/bin/env julia
 
#Start Test Script
using EKI
if VERSION < v"0.7"
    using Base.Test
else
    using Test
    using Random
    using Statistics
end
 
# Run tests
 
@testset "EKI Tests" begin
    @testset "CovarianceOperator Tests" begin
        x = randn(500)
        X = randn(20,500)
        C = cov(X,X)
        A = EKI.CovarianceOperator(X)
        y = C*x
        Y = randn(20,300)
        A2 = EKI.CrossCovarianceOperator(X,Y)
        C2 = cov(X,Y)
        x2 = randn(300)
        y2 = C2*x2
        y1p5 = C2'*x
        @test isapprox(C, convert(Array{Float64}, A))
        @test isapprox(y, A*x)
        @test isapprox(convert(Array{Float64}, A), convert(Array{Float64}, A'))
        @test_broken isapprox(C2, convert(Array{Float64}, A2))
        @test isapprox(y2, A2*x2)
        @test isapprox(y1p5, A2'*x)
        @test_broken isapprox(x'*B*x2, x2'*B'*x)
    end
    @testset "EKI Tests" begin
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
            xe = eki([yt], σ, η, J, N, prior, gmap; verbosity=1,ρ=0.5,ζ=2.0) 
            @test sqrt((yt-G*xe)'*((y-G*xe))/σ^2) <= 2.0*η 
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
            xe = eki(yt, σ, η, J, N, prior, gmap; verbosity=1,ρ=0.5,ζ=2.0, batched=true, batches=3, batch_off=20) 
            ye = [G1*xe,G2*xe,G3*xe,G4*xe]
            @test sqrt((vcat(yt...)-vcat(ye...))'*((vcat(y...)-vcat(ye...)))/σ^2) <= 2.0*η 
        end    
    end
end
