using LPVSpectral
using Test, LinearAlgebra, Statistics, Random
using Plots, DSP
Random.seed!(0)
# write your own tests here
function generate_signal(f,w,N, modphase=false)
    x = sort(10rand(N)) # Sample points
    v = range(0, stop=1, length=N) # Scheduling variable

    # generate output signal
    # phase_matrix
    dependence_matrix = Float64[f[(i-1)%length(f)+1].(v) for v in v, i in eachindex(w)] # N x nw
    frequency_matrix = [cos(w*x -0.5modphase*(dependence_matrix[i,j])) for (i,x) in enumerate(x), (j,w) in enumerate(w)] # N x nw
    y = sum(dependence_matrix.*frequency_matrix,dims=2)[:] # Sum over all frequencies
    y += 0.1randn(size(y))
    y,v,x,frequency_matrix, dependence_matrix
end

@testset "LPVSpectral" begin
    @info "Testing LPVSpectral"

    @testset "Mel" begin
        @info "Testing Mel"

        M = mel(1,256)
        @test size(M,1) == 128
        @test size(M,2) == 256÷2+1
        M = mel(1000,256, fmin=100)
        sum(M[:,1:26]) == 0

        y = randn(1000)
        M = melspectrogram(y)
        @test length(freq(M)) == 128
        @test size(M.power) == (128,14)
        @test length(time(M)) == 14
        plot(M)

        M = mfcc(y)
        @test length(freq(M)) == 20
        @test size(M.mfcc) == (20,14)
        @test length(time(M)) == 14



    end

    @testset "LPV methods" begin
        @info "testing LPV methods"

        N      = 500 # Number of training data points
        f      = [v->2v^2, v->2/(5v+1), v->3exp(-10*(v-0.5)^2),] # Functional dependences on the scheduling variable
        w      = 2pi*[2,10,20] # Frequency vector
        w_test = 2π*collect(2:2:25)
        Y,V,X,frequency_matrix, dependence_matrix = generate_signal(f,w,N,true)

        λ      = 0.02 # Regularization parmater
        normal = true # Use normalized basis functions
        Nv     = 50 # Number of basis functions

        se = ls_spectral_lpv(Y,X,V,w_test,Nv; λ = λ, normalize = normal) # Perform LPV spectral estimation
        @show plot(se, phase=true)
        windowpsd = ls_windowpsd_lpv(Y,X,V,w_test,Nv; λ = λ, normalize = normal)

        spectrum_lpv   = psd(se)

        si = sortperm(spectrum_lpv[:],rev=true)
        @test Set(si[1:3]) == Set([1,5,10])

        si = sortperm(windowpsd[:],rev=true)
        @test Set(si[1:3]) == Set([1,5,10])
    end

    @testset "detrend" begin
        @info "testing detrend"
        tre = [1,2,3]
        @test detrend(tre) == [-1,0,1]
        @test tre == [1,2,3] # Shouldn't have changed
        detrend!(tre)
        @test tre == [-1,0,1] # Should have changed
    end


    ## Test ComplexNormal
    @testset "Complex Normal" begin
        @info "testing Complex Normal"
        n  = 10
        n2 = 5
        a  = randn(n)
        A  = randn(n,n)
        A  = A'A + I
        b  = randn(n2)
        B  = randn(n2,n2)
        B  = B'B + I
        X  = randn(n,n2)
        Y  = randn(n,n2)
        cn = ComplexNormal(X,Y)
        @test size(cn.m) == (n2,)
        @test size(cn.Γ) == (n2,n2)
        @test size(cn.C) == (n2,n2)
        # affine_transform(cn, B, b)
        pdf(cn,b)
        pdf(cn,im*b)

        @test isa(cn_Vxx(A,A), Cholesky)
        @test isa(cn_fVxx(A,A), Matrix)
        @test issymmetric(cn_fVxx(A,A))
        cn_V(A,0.1A)

        cn  = ComplexNormal(a,A)
        cn  = ComplexNormal(im*b,A)

        A   = randn(4,4);
        A   = A'A
        x   = randn(1000,3)
        y   = randn(1000,3)
        cn  = ComplexNormal(x,y)
        z   = rand(cn,1000000);
        cn2 = ComplexNormal(z)
        @test norm(Matrix(cn.Γ)-Matrix(cn2.Γ)) < 0.01
        @test norm(Matrix(cn.C)-Matrix(cn2.C)) < 0.01

    end

    @testset "ls methods" begin
        @info "testing ls methods"
        T = 1000
        t = 0:T-1
        f = LPVSpectral.default_freqs(t)
        @test f[1] == 0
        @test f[end] == 0.5-1/length(t)/2
        @test length(f) == T÷2

        # f2 = LPVSpectral.default_freqs(t,10)
        # @test f2[1] == 0
        # @test f2[end] == 0.5-1/length(t)/2
        # @test length(f2) == T÷2÷10

        @test LPVSpectral.check_freq(f) == 1
        @test_throws ArgumentError LPVSpectral.check_freq([1,0,2])
        A, z = LPVSpectral.get_fourier_regressor(t,f)
        @test size(A) == (T,2length(f)-1)

        Base.isapprox(t1::Tuple{Float64,Int64}, t2::Tuple{Float64,Int64}; atol) = all(t -> isapprox(t[1],t[2],atol=atol), zip(t1,t2))
        y = sin.(t)
        x,_ = ls_spectral(y,t)
        @test findmax(abs.(x)) ≈ (1.0, 160) atol=1e-4

        W = ones(length(y))
        x,_ = ls_spectral(y,t,f,W)
        @test findmax(abs.(x)) ≈ (1.0, 160) atol=1e-4

        x,_ = tls_spectral(y,t)
        @test findmax(abs.(x)) ≈ (1.0, 160) atol=1e-4

        x,_ = ls_windowpsd(y,t,noverlap=0)
        @test findmax(x) ≈ (1.0, 160) atol=0.01

        x,_ = ls_windowpsd(y,t,nw=16,noverlap=0)
        @test findmax(abs.(x)) ≈ (1.0, 160) atol=0.15

        x,_ = ls_windowcsd(y,y,t,noverlap=0)
        @test findmax(abs.(x)) ≈ (1.0, 160) atol=0.12


        x,_ = ls_cohere(y,y,t)
        @test all(x .== 1)

        x,_ = ls_cohere(y,y .+ 5randn.(),t,nw=8,noverlap=-1)
        @show mean(x)
        @test mean(x) < 0.9

    end

    @testset "plots" begin
        @info "testing plots"

        y = randn(1000)
        @show plot(periodogram(y))
        @show plot(periodogram(filtfilt(ones(4), [4], y)))
        @show plot(welch_pgram(y))


    end

    @testset "Lasso" begin
        @info "Testing Lasso"

        include("test_lasso.jl")
    end

end
