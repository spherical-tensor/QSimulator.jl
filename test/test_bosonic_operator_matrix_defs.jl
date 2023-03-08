using Test, QSimulator
using LinearAlgebra: I
using SparseArrays

@testset "Single Body Operators" begin
    import QSimulator: raise!, lower!, setdiag!
    d=10
    m = spzeros(Complex, d,d)
    # raise!, lower!, setdiag! Definitions
    @testset "raise!,lower!,setdiag!" begin 
        i = 5
        c = 3+4im # allow complex scaling
        raise!(m, c, i)
        @test m[i+1, i] == sqrt(i)*c
        @test length(m.nzval) == 1
        lower!(m, c, i)
        @test m[i-1, i] == sqrt(i-1)*c
        @test length(m.nzval) == 2
        setdiag!(m, c, i)
        @test m[i, i] == c
        @test length(m.nzval) == 3
    end

    @testset "Bosonic Operator Consistency" begin
        d= 10
        ϕ = rand()
        scale = rand()
        # check definition consistency, lowering† == raising
        adag = raising(d, ϕ, scale)
        a = lowering(d, ϕ, scale)
        @test raising(d) == lowering(d)'
        @test a' ≈ adag
        c(x,y) = x*y - y*x
        # apart from the last element, all should be i * scale
        @test c(a, adag)[1:d-1,1:d-1] ≈ scale^2*I(d-1)
    end
end
