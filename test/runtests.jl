using LPVSpectral
using Test, LinearAlgebra, Statistics, Random, StatsBase
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



    @testset "Windows" begin
        @info "Testing Windows"

        y = 1:100
        t = 1:100
        W = Windows2(y,t,10,0)
        @test length(W) == 10
        @test first(W) == (1:10,1:10)
        res = mapwindows(W) do (y,t)
            -y
        end
        @test res == -1:-1:-100

        W = Windows2(y,t,10,1)
        @test length(W) == 11
        cW = collect(W)
        @test cW[1] == (1:10,1:10)
        @test cW[2] == (10:19,10:19)

        res = mapwindows(W) do (y,t)
            -y
        end
        @test res == -1:-1:-100

        y = 1:100
        t = 1:100
        W = Windows3(y,t,t,10,0)
        @test length(W) == 10
        @test first(W) == (1:10,1:10,1:10)

        W = Windows3(y,t,t,10,1)
        @test length(W) == 11
        cW = collect(W)
        @test cW[1] == (1:10,1:10,1:10)
        @test cW[2] == (10:19,10:19,10:19)



    end

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
        T = 100
        t = 0:0.1:T-0.1
        f = LPVSpectral.default_freqs(t)
        @test f[1] == 0
        @test f[end] == 5
        @test length(f) == 10T÷2+1

        # f2 = LPVSpectral.default_freqs(t,10)
        # @test f2[1] == 0
        # @test f2[end] == 0.5-1/length(t)/2
        # @test length(f2) == T÷2÷10

        @test LPVSpectral.check_freq(f) == 1
        @test_throws ArgumentError LPVSpectral.check_freq([1,0,2])
        A, z = LPVSpectral.get_fourier_regressor(t,f)
        @test size(A) == (10T,2length(f)-1)

        Base.isapprox(t1::Tuple{Float64,Int64}, t2::Tuple{Float64,Int64}; atol) = all(t -> isapprox(t[1],t[2],atol=atol), zip(t1,t2))
        y = sin.(2pi .* t)
        x,_ = ls_spectral(y,t)
        @test findmax(abs.(x)) ≈ (1.0, 101) atol=1e-4

        W = ones(length(y))
        x,_ = ls_spectral(y,t,f,W)
        @test findmax(abs.(x)) ≈ (1.0, 101) atol=1e-4

        x,freqs = tls_spectral(y,t)
        @test findmax(abs.(x)) ≈ (1.0, 101) atol=1e-4

        x,freqs = ls_windowpsd(y,t,noverlap=0)
        @test findmax(x)[2] ==  13

        x,freqs = ls_windowpsd(y,t,nw=16,noverlap=0)
        @test findmax(abs.(x))[2] ==  7

        x,freqs = ls_windowcsd(y,y,t,noverlap=0)
        @test findmax(abs.(x)) ≈ (1.0, 11) atol=1e-4


        x,_ = ls_cohere(y,y,t)
        @test all(x .== 1)

        x,freqs = ls_cohere(y,y .+ 0.5 .*randn.(),t,nw=8,noverlap=-1)
        @show mean(x)
        @test findmax(x) ≈ (1.0, 14) atol=0.15
        @test mean(x) < 0.25

    end

    @testset "plots" begin
        @info "testing plots"

        y = randn(1000)
        @show plot(periodogram(y))
        @show plot(spectrogram(y))
        @show plot(melspectrogram(y))
        @show plot(periodogram(filtfilt(ones(4), [4], y)))
        @show plot(welch_pgram(y))


    end

    @testset "Lasso" begin
        @info "Testing Lasso"
        include("test_lasso.jl")
    end


    @testset "autocov" begin
        @info "Testing autocov"


        y = repeat([1.,0.,-1.],100)
        τ,acf = autocov(1:length(y), y, Inf)
        acf0 = autocov(y, demean=false)
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.01
        @test norm(acfh - acf0) < 0.01


        τ,acf = autocor(1:length(y), y, Inf)
        acf0 = autocor(y, demean=false)
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.01
        @test norm(acfh - acf0) < 0.1

        y = randn(100)
        τ,acf = autocor(1:length(y), y, Inf)
        acf0 = autocor(y, demean=false)
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.01
        @test norm(acfh - acf0) < 0.1



        y = randn(10)
        τ,acf = autocov(1:length(y), y, Inf)
        acf0 = autocov(y, demean=false)
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.01
        @test norm(acfh - acf0) < 0.2

        y = randn(10)
        τ,acf = autocor(1:length(y), y, Inf)
        acf0 = autocor(y, demean=false)
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.03
        @test norm(acfh - acf0) < 0.2


        y = [randn(10) for _ in 1:10]
        t = reshape(1:100,10,10)
        t = collect(eachcol(t))
        τ,acf = autocov(t, y, Inf)
        acf0 = mean(autocov.(y, demean=false))
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.01
        @test norm(acfh - acf0) < 0.2

        τ,acf = autocor(t, y, Inf)
        acf0 = mean(autocor.(y, demean=false))
        acfh = [mean(acf[τ.==i]) for i = 0:length(acf0)-1]
        @test acfh ≈ acf0 rtol=0.01
        @test norm(acfh - acf0) < 0.2

        # See what happens if one series is contant
        y = zeros(10)
        τ,acf = autocor(1:10, y, Inf)
        @test all(acf .== 1)

        using LPVSpectral: _autocov, _autocor, isequidistant

        @test isequidistant(1:5)
        @test isequidistant(1:2:10)
        @test !isequidistant(reverse(1:2:10))
        @test isequidistant(collect(1:5))
        @test isequidistant(collect(1:2:10))
        @test isequidistant(collect(1:0.33:10))


        res = map(1:10) do _
            t = 100rand(100)
            @test !isequidistant(t)
            t0 = 0:99
            y = sin.(0.05 .* t)
            y0 = sin.(0.05 .* t0)
            τ0, acf0 = autocor(t0, y0, Inf, normalize=true)
            τ,acf = autocor(t, y, Inf)
            acff = filtfilt(ones(200),[200], acf)
            @test count(τ .== 0) == length(y)
            # plot(τ[1:10:end],acf[1:10:end])
            # plot!(τ0[1:10:end],acf0[1:10:end])
            # plot!(τ0[1:10:end],acff[1:10:end])
            mean(abs2,acf0-acff) < 0.05
        end
        @test mean(res) > 0.7

        
        res = map(1:10) do _
            t = 100rand(100)
            @test !isequidistant(t)
            t0 = 0:99
            y = sin.(0.05 .* t)
            y0 = sin.(0.05 .* t0)
            τ0, acf0 = autocov(t0, y0, Inf, normalize=true)
            τ,acf = autocov(t, y, Inf)
            acff = filtfilt(ones(200),[200], acf)
            @test count(τ .== 0) == length(y)
            # plot(τ[1:10:end],acf[1:10:end])
            # plot!(τ0[1:10:end],acf0[1:10:end])
            # plot!(τ0[1:10:end],acff[1:10:end])
            mean(abs2,acf0-acff) < 0.025
        end
        @test mean(res) > 0.7

    end

end
