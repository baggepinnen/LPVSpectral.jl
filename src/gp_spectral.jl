SE(z1,z2,w,s) = exp(-0.5/s^2*(z1[2]-z2[2]).^2)

function normalizedSE(z1,z2,w,s)
    r = Float64[exp(-0.5/s^2*(z1[2]-z2[2]).^2) for z2 in z2]
    r ./=sum(r)
end
fourier(z1,z2,w,s) = exp(-im*w*(z1[1]))

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
        K(z1,z2,w) = vec(vec(rw.rw(z1, z2, w,rw.params))*rv.rv(z1,z2,w,rv.params)')
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
    R = Array(Complex128,length(z1),length(z2)*length(w))
    for (i,z1) in enumerate(z1)
        R[i,:] = K(z1,z2,w)
    end
    R
end


reshape_params(s::GPspectum, args...) = reshape(s.params, length(s.w), length(s.Y))


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
    Σn = σn*eye(2np) # Noise covariance
    A0 = _A(Z,Z,w,K)    # Raw regressor matrix

    if false # Use realified arithmetics
        A1 = [real(A0) imag(A0); Σn]       # Helper matrix for numerically robust ridge regression
        # A1 = [real(A0) imag(A0)]       # Helper matrix for numerically robust ridge regression
        A2 = factorize(A1)    # factorized object which is fast to invert
        function bs(b)
            x = A2\[b;zeros(2np)] # Backslash function using the factorized object and numerically robust ridge regression
            complex(x[1:np], x[np+1,end])
        end
    else
        A1 = [A0; σn*eye(np)]       # Helper matrix for numerically robust ridge regression
        A2 = factorize(A1)    # factorized object which is fast to invert
        function bs(b)
            x = A2\[b;zeros(np)] # Backslash function using the factorized object and numerically robust ridge regression
        end
    end
    a(z) = K(z,Z,w) # Covariance between all training inputs and z

    params = bs(Y) # Estimate the parameters. This is now (A'A+σ²I)\A'Y
    mD(z::Vector{Float64}) =  a(z)'params
    mD(z) = _A(z,Z,w,K)*params
    KD(z,zp) = _A(z,zp,w,K) - a(z)*bs(a(zp)')
    A1 = [real(A0) imag(A0)] # TODO: should Σn be appended here?
    Σ = σn^2*inv(A1'A1)
    dist = ComplexNormal(params,Σ)

    return GPspectum(opts,Y,X,V,w,mD,KD,params)
end

## Plotting functionality ---------------------------------------
import Plots.plot

Plots.plot(s::GPspectum) = plot(s,:y)

function Plots.plot(s::GPspectum, types...;  normalization=:sum, normdim=:freq, dims=3)
    for t in types
        t ∈ [:y, :Y, :outout] && plot_output(s)
        t ∈ [:spectrum] && plot_spectrum(s)
        t ∈ [:schedfunc] && plot_schedfunc(s, normalization=normalization, normdim=normdim, dims=dims)
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

function plot_schedfunc(s::GPspectum; normalization=:none, normdim=:vel, dims=3)
    Kt(v) = s.opts.rv.rv([0,v],Z,s.w,s.opts.rv.params)
    return plot_schedfunc(s.params,s.V,s.w,Kt; normalization=normalization, normdim=normdim, dims=dims)
end
