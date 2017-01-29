module TestLPV
using LPVSpectral
using Base.Test
# write your own tests here

function generate_signal(f,w,N, modphase=false)
    x = 2sort(rand(N)) # Sample points
    x = linspace(0,2,N)
    v = linspace(0,1,N) # Scheduling variable

    # generate output signal
    # phase_matrix
    dependence_matrix = Float64[f[i](v) for v in v, i in eachindex(f)] # N x nw
    frequency_matrix = [sin(w*x + modphase*dependence_matrix[i,j]) for (i,x) in enumerate(x), (j,w) in enumerate(w)] # N x nw
    y = sum(dependence_matrix.*frequency_matrix,2) # Sum over all frequencies
    y,v,x,frequency_matrix, dependence_matrix
end


N = 500 # Number of training data points
f = [v->2v^2, v->2/(5v+1), v->3exp(-10*(v-0.5)^2),] # Functional dependences on the scheduling variable
w = 2pi*[2,10,50] # Frequency vector
Y,V,X,frequency_matrix, dependence_matrix = generate_signal(f,w,N)

λ = 0.02        # Regularization parmater
normal = true   # Use normalized basis functions
Nv = 50         # Number of basis functions

se = ls_spectralext(Y,X,V,w_test,Nv; λ = λ, normalize = normal) # Perform LPV spectral estimation

Z = [Float64[X[i], V[i]] for i in eachindex(X)]
z2 = Z
z1 = Z[1]

sigma = (maximum(V)-minimum(V))/(0.5*N)*100
σn = 0.05

# opts = GPspectralOpts(σn, sigma)
#
# # s = GP_spectral(Y,X,V,w, opts)
# R = Array(Complex128,N,length(z2),length(w))
# @test size(LPVSpectral.normalizedSE(z1,z2,w[1],sigma)) == size(z2)
# @test size(LPVSpectral.fourier(z1,z2,w,sigma)) == size(w)
# @test length(opts.K(z1,z2,w)) == length(Z)*length(w)
#
# @test size(LPVSpectral._A(z2, z2, w, opts.K)) == (N,N*length(w))





## Test ComplexNormal
println("Testing Complex Normal")
n = 10
n2 = 5
a = randn(n)
A = randn(n,n)
A = A'A + eye(n)
b = randn(n2)
B = randn(n2,n2)
B = B'B + eye(n2)
X = randn(n,n2)
Y = randn(n,n2)
cn = ComplexNormal(X,Y)
@test size(cn.m) == (n2,)
@test size(cn.Γ) == (n2,n2)
@test size(cn.C) == (n2,n2)
# affine_transform(cn, B, b)
pdf(cn,b)
pdf(cn,im*b)

@test isa(cn_Vxx(A,A), Base.LinAlg.Cholesky)
@test isa(cn_fVxx(A,A), Matrix)
@test issymmetric(cn_fVxx(A,A))
cn_V(A,0.1A)

cn = ComplexNormal(a,A)
cn = ComplexNormal(im*b,A)

A = randn(4,4);
A = A'A
x = randn(1000,3)
y = randn(1000,3)
cn = ComplexNormal(x,y)
z = rand(cn,1000000);
cn2 = ComplexNormal(z)
@test vecnorm(full(cn.Γ)-full(cn2.Γ)) < 0.01
@test vecnorm(full(cn.C)-full(cn2.C)) < 0.01

println("Done")


end
