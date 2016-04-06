SE(z1,z2,w,s) = exp(-0.5/s^2*(z1[2]-z2[2])^2)

function normalizedSE(z1,z2,w::Real,s)
    r = Float64[exp(-0.5/s^2*(z1[2]-z2[2])^2) for z2 in z2]
    r ./=sum(r)
end
fourier(z1,z2,w::Real,s) = Complex128[exp(-im*w*(z1[1]-z2[1])) for z2 in z2]

manhattan(x) = real(x)+imag(x)

type GPcov
    rv
    params
end

type GPfreq
    rw
    params
end

type GPspectralOpts
    σn
    rv::GPcov
    rw::GPfreq
    K # Combined covariance/kernel function)
    function GPspectralOpts(σn, rv, rw)
        K(z1,z2,w) = rv.rv(z1,z2,w,rv.params).*rw.rw(z1, z2, w,rw.params)
        new(σn, rv, rw, K)
    end
end

GPspectralOpts(σn, s; rv=GPcov(normalizedSE,s), rw=GPfreq(fourier,0)) = GPspectralOpts(σn, rv, rw)

type GPspectum
    opts::GPspectralOpts
    Y
    X
    V
    w
    m
    K
    params
end

augment_state(X,V) = [Float64[X[i], V[i]] for i in eachindex(X)]
augment_state(s::GPspectum) = augment_state(s.X,s.V)

function _A(z1, z2, w, K)
    N = length(z1)
    R = Array(Complex128,N,length(z2),length(w))
    for (i,z1) in enumerate(z1), (j,w) in enumerate(w)
        R[i,:,j] = K(z1,z2,w)
    end

    reshape(R,N, round(Int,prod(size(R))/N))
end

function _A(z1::Array{Float64}, z2, w, K)
    R = Array(Complex128,length(z2),length(w))
    for (j,w) in enumerate(w)
        R[:,j] = K(z1,z2,w)
    end
    R[:]'
end

""" Returns params as a [nω × N] matrix"""
reshape_params(s::GPspectum) = reshape(s.params, length(s.w), length(s.Y))


function param_cov(A0,opts)
    Σi = A0'A0 + opts.σn*eye(size(A0,2)) # Inverse of parameter covariance, without σe
    Σ = opts.σn^2*inv(Σi) # Full parameter covariance matrix
end

"""
``

`Y` is the signal to be decomposed

`X` is a vector of sampling points

`V` is a vector with the scheduling signal

`w` is a vector of frequencies to decompose the signal at

`opts::GPspectralOpts` is an object with options, defaults to Gaussian covariance and complex sinusoidal basis functions in the frequency domain

"""
function GP_spectral(Y,X,V,w,
    opts::GPspectralOpts = GPspectralOpts(0,maximum(V)-minimum(V))/(0.5*length(Y)))

    N = length(Y)
    Z = augment_state(X,V) # Augmented state
    K = opts.K # Combined covariance/kernel function
    σn = opts.σn

    np = length(w)*N # Number of parameters
    Σn = σn*eye(np) # Noise covariance
    A0 = _A(Z,Z,w,K)    # Raw regressor matrix

    A1 = [A0; Σn]       # Helper matrix for numerically robust ridge regression
    A2 = svdfact(A1)    # SVD object which is fast to invert

    bs(b) = A2\[b;zeros(np)] # Backslash function using the SVD object and numerically robust ridge regression
    a(z) = _A(z,Z,w,K) # Covariance between all training inputs and z

    params = bs(Y) # Estimate the parameters. This is now (A'A+σI)\A'Y
    mD(z) =  a(z)*params
    KD(z,zp) = _A(z,zp,w,K) - a(z)*bs(a(zp)')


    return GPspectum(opts,Y,X,V,w,mD,KD,params)
end

## Plotting functionality ---------------------------------------
function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end

import Plots.plot

Plots.plot(s::GPspectum) = plot(s,:y)

function Plots.plot(s::GPspectum, types...;  normalization=:sum, normdim=:freq)
    for t in types
        t ∈ [:y, :Y, :outout] && plot_output(s)
        t ∈ [:spectrum] && plot_spectrum(s)
        t ∈ [:schedfunc] && plot_schedfunc(s, normalization=normalization, normdim=normdim)
    end
end


function plot_output(s)
    Z = augment_state(s)
    A0 = _A(Z,Z,s.w,s.opts.K)
    Σ = param_cov(A0,s.opts)
    covs = manhattan(sqrt(diag(A0*Σ*A0')))
    Yhat = manhattan(s.m(Z)[:])
    plot([s.Y Yhat Yhat+2covs Yhat-2covs], lab=["\$y\$" "\$ŷ\$"], c=[:red :blue :cyan :cyan])
end

plot_spectrum(s) = 0

function plot_schedfunc(s; normalization=:max, normdim=:vel)
    Z = LPVSpectral.augment_state(s)
    x = manhattan(LPVSpectral.reshape_params(s)) # [nω × N]
    Nf = length(s.w)
    Nv = 50
    N = length(s.Y)
    ax  = abs(x)
    px  = angle(x)
    rv = s.opts.rv.rv


    fg,vg = LPVSpectral.meshgrid(s.w,linspace(minimum(s.V),maximum(s.V),Nv))
    A = zeros(size(fg))
    P = zeros(size(fg))

    for j = 1:Nv, i = 1:Nf # freqs
        # (z1,z2,w,s)
        r = Float64[rv([0,vg[i,j]],Z,s.w,s.opts.rv.params) for Z in Z]
        A[i,j] = (ax[i,:]*r)[1]
        P[i,j] = (px[i,:]*r)[1]
    end

    nd = normdim == :freq ? 1 : 2
    normalizer = 1
    if normalization == :sum
        normalizer =   sum(A, nd)/size(A,nd)

    elseif normalization == :max
        normalizer =   maximum(A, nd)
    end
    A = A./normalizer

    plot3d()
    for i = 1:Nf
        plot3d!(fg[i,:]'[:],vg[i,:]'[:],A[i,:]'[:])
    end
    plot3d!(ylabel="\$v\$", xlabel="\$ω\$")#, zlabel="\$f(v)\$")


    # TODO: plot confidence intervals for these estimates

end
