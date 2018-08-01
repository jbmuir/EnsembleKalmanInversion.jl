#!/usr/bin/env julia
 
#Start Test Script
using EKI
using Base.Test
 
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
    # @testset "Basic EKI Tests" begin
    #     @testset "Basic Linear Test" begin
    #     #This should certainly be a set of "convergence" tests
    #         G = randn(2500,100)
    #         x = randn(100)
    #         σ = 0.1
    #         yt = G*x
    #         y = yt .+ σ * randn(2500)
    #         η = sqrt((y-yt)'*(y-yt)/σ^2)
    #         prior() = randn(100)
    #         gmap(q) = G*q 
    #         J = 100
    #         N = 100
    #         xe = eki(y, σ, η, J, N, prior, gmap; verbosity=1) 
    #         @test sqrt((yt-G*xe)'*((y-G*xe))/σ^2) <= η 
    #     end
    # end
    @testset "Lowmem EKI Tests" begin
    @testset "Lowmem Linear Test" begin
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
        xe = eki(yt, σ, η, J, N, prior, gmap; verbosity=1, lowmem=true, γ0=1e-9) 
        @test sqrt((yt-G*xe)'*((y-G*xe))/σ^2) <= η 
    end    end
end
