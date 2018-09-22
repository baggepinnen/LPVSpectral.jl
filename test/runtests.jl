module TestLPV
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


N      = 500 # Number of training data points
f      = [v->2v^2, v->2/(5v+1), v->3exp(-10*(v-0.5)^2),] # Functional dependences on the scheduling variable
w      = 2pi*[2,10,20] # Frequency vector
w_test = 2π*collect(2:2:25)
Y,V,X,frequency_matrix, dependence_matrix = generate_signal(f,w,N,true)

λ      = 0.02 # Regularization parmater
normal = true # Use normalized basis functions
Nv     = 50 # Number of basis functions

se = ls_spectral_lpv(Y,X,V,w_test,Nv; λ = λ, normalize = normal) # Perform LPV spectral estimation
windowpsd = ls_windowpsd_lpv(Y,X,V,w_test,Nv; λ = λ, normalize = normal)

spectrum_lpv   = psd(se)

si = sortperm(spectrum_lpv[:],rev=true)
@test Set(si[1:3]) == Set([1,5,10])

si = sortperm(windowpsd[:],rev=true)
@test Set(si[1:3]) == Set([1,5,10])

tre = [1,2,3]
@test detrend(tre) == [-1,0,1]
@test tre == [1,2,3] # Shouldn't have changed
detrend!(tre)
@test tre == [-1,0,1] # Should have changed

## Test ComplexNormal
println("Testing Complex Normal")
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

println("Done")


end
