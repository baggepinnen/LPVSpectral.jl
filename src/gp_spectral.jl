SE(z1,z2,w,s) = exp(-0.5/s^2*(z1[2]-z2[2])^2)
fourier(z1,z2,w,s) = exp(-im*w*(z1[1]-z2[1]))
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
    GPspectralOpts(σn, rv, rw) = new(σn, rv, rw, (z1,z2,w) -> rv.rv(z1,z2,w,rv.params)*rw.rw(z1, z2, w,rw.params))
end

GPspectralOpts(σn, s; rv=GPcov(SE,s), rw=GPfreq(fourier,0)) = GPspectralOpts(σn, rv, rw)

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
    R = Complex128[K(z1,z2,w) for z1 in z1, z2 in z2, w in w]
    N = length(z1)
    reshape(R,N, round(Int,prod(size(R))/N))
end

function _A(z1::Array{Float64}, z2, w, K)
    R = Complex128[K(z1,z2,w) for z2 in z2, w in w][:]'
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

function Plots.plot(s::GPspectum, types...)
    for t in types
        t ∈ [:y, :Y, :outout] && plot_output(s)
        t ∈ [:spectrum] && plot_spectrum(s)
        t ∈ [:schedfunc] && plot_schedfunc(s)
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

function plot_schedfunc(s)
    Z = augment_state(s)
    x = reshape_params(s) # [nω × N]
    Nf = length(s.w)
    N = length(s.Y)
    ax  = abs(x)
    px  = angle(x)
    rv = s.opts.rv.rv

    fg,vg = meshgrid(s.w,linspace(minimum(s.V),maximum(s.V),20))
    F = zeros(size(fg))
    P = zeros(size(fg))

    for j = 1:N, i = 1:Nf # freqs
        # (z1,z2,w,s)
            F[j,i] = ax[i,:]*rv([0,vg[j,i]],Z)
            P[j,i] = px[i,:]*rv([0,vg[j,i]],Z)
    end
    #
    #
    #     display("Spectral estimate ")
    #     if false # Normalize over velocities (max)
    #         F = F./repmat(max(F,2),1,Nf)
    #         display("normalized so max over freqencies is 1 for each velocity")
    #     end
    #
    #     if true # Normalize over velocities (sum) (tycks vara den bästa)
    #         F = F./repmat(sum(F,2),1,Nf)
    #         display("normalized so sum over freqencies is 1 for each velocity")
    #     end
    #
    #     if false # Normalize over frequencies
    #         F = F./repmat(max(F,1),20,1)
    #         display("normalized so max over velocities is 1 for each freqency")
    #     end
    #
    #     if false # Normalize over frequencies (sum)
    #         F = F./repmat(sum(F,1),20,1)
    #         display("normalized so sum over velocities is 1 for each freqency")
    #     end
    #
    #     if plotres
    #         if false
    #             figure,
    #             waterfall(fg',vg',F')
    #             zlabel("Amplitude")
    #             xlabel("Frequency")
    #             ylabel("Velocity [rad/s]")
    #             alpha(0.2)
    #             # %         set(gca,"zscale","log")
    #         end

du höll på att fixa det sista i denna funktionen :D







end
