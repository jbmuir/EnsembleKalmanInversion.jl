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
        @test isapprox(C, convert(Array{Float64}, A))
        @test isapprox(y, A*x)
        @test isapprox(convert(Array{Float64}, A), convert(Array{Float64}, A'))
        Y = randn(20,300)
        B = EKI.CrossCovarianceOperator(X,Y)
    end
    @testset "Basic EKI Tests" begin
        @test 2==2
    end
    @testset "Lowmem EKI Tests" begin
        @test 2==2
    end
end