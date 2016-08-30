
"""`ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))`"""
function ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y); λ=0)
    N = length(y)
    Nf = length(f)
    A = [exp(2π*f[fn]*t[n]) for n = 1:N, fn = 1:Nf]
    x = real_complex_bs(A,b,λ)
    print_with_color(:magenta,"$(round(cond(A'A),2))\n")
end


"""`ls_spectral(y,t,f,W::AbstractVector)`"""
function ls_spectral(y,t,f,W::AbstractVector)
    N = length(y)
    Nf = length(f)
    A = zeros(N,Nf)
    for n = 1:N, fn=1:Nf
        phi = 2π*f[fn]*t[n]
        A[n,fn] = cos(phi)
        A[n,fn+Nf] = -sin(phi)
    end

    W = diagm([W;W])
    x   = (A'W*A)\A'W*y
    print_with_color(:magenta,"$(round(cond(A'*W*A),2))\n")
    x = complex(x[1:Nf], x[Nf+1:end])
end


"""`tls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))`"""
function tls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))
    N = length(y)
    Nf = length(f)
    A = zeros(N,2Nf)
    for n = 1:N, fn=1:Nf
        phi = 2π*f[fn]*t[n]
        A[n,fn] = cos(phi)
        A[n,fn+Nf] = -sin(phi)

    end
    AA = [A y]
    U,S,V = svd(AA)
    m,n,p = N,2Nf,1
    V21 = V[1:n,n+1:end]
    V22 = V[n+1:end,n+1:end]
    x = -V21/V22
    # x = x[1:Nf] + 1im*x[Nf+1:end]
    print_with_color(:blue,"$(round(cond(AA'AA),2))\n")
    x = complex(x[1:Nf], x[Nf+1:end])
end


"""`ls_windowpsd(y,t,freqs, nw = 10, noverlap = 0)`"""
function ls_windowpsd(y,t,freqs, nw = 10, noverlap = 0)
    N       = length(y)
    dpw     = floor(Int64,N/nw)
    inds    = 1:dpw
    if noverlap == 0
        noverlap = round(Int64,dpw/2)
    end

    function hanningjitter(t)
        n = length(t)
        t = t-t[1]
        t *= (n-1)/t[end]
        [0.5*(1 - cos(2π*k/(n-1))) for k=t]
    end
    W     = DSP.hanning(dpw)
    S     = 0.
    for j = 1:nw
        # W     = hanningjitter(t[inds])
        x     = ls_spectral(y[inds],t[inds],freqs,W)
        inds  = inds + (dpw - noverlap)
        # Power spectrum
        S += abs(x).^2
    end
    return S
end

"""`ls_windowcsd(y,u,t,freqs, nw = 10, noverlap = 0)`"""
function ls_windowcsd(y,u,t,freqs, nw = 10, noverlap = 0)
    N       = length(y)
    dpw     = floor(Int64,N/nw)
    if noverlap == 0
        noverlap = round(Int64,dpw/2)
    end
    inds    = 1:dpw
    S       = 0.
    W       = hanning(dpw)
    for j = 1:nw
        xy      = ls_spectral(y[inds],t[inds],freqs,W)
        xu      = ls_spectral(u[inds],t[inds],freqs,W)
        inds  = inds + (dpw - noverlap)
        # Cross spectrum
        S += xy.*conj(xu)
    end
    return S
end


# function lscohere(y,u,t,freqs, nw = 10, noverlap = 0)
#         Syu     = lswindowcsd(y,u,t,freqs, nw, noverlap)
#         Syy     = lswindowpsd(y,  t,freqs, nw, noverlap)
#         Suu     = lswindowpsd(u,  t,freqs, nw, noverlap)
#         Sch     = (abs(Syu).^2)./(Suu.*Syy);
# end

"""`ls_cohere(y,u,t,freqs, nw = 10, noverlap = 0)`"""
function ls_cohere(y,u,t,freqs, nw = 10, noverlap = 0)
    N       = length(y)
    dpw     = floor(Int64,N/nw)
    if noverlap == 0
        noverlap = round(Int64,dpw/2)
    end
    inds    = 1:dpw
    Syy       = 0.
    Suu       = 0.
    Syu       = Complex128(0.0)
    W         = hanning(dpw)
    for j = 1:nw
        xy      = ls_spectral(y[inds],t[inds],freqs,W)
        xu      = ls_spectral(u[inds],t[inds],freqs,W)
        inds  = inds + (dpw - noverlap)
        # Cross spectrum
        Syu += xy.*conj(xu)
        Syy += abs(xy).^2
        Suu += abs(xu).^2
    end
    Sch     = (abs(Syu).^2)./(Suu.*Syy);
    return Sch
end

@inline _K(V,vc,gamma) = exp(-gamma*(V-vc).^2)

@inline function _K_norm(V,vc,gamma)
    r = _K(V,vc,gamma)
    r ./=sum(r)
end

@inline _Kcoulomb(V,vc,gamma) = _K(V,vc,gamma).*(sign(V) .== sign(vc))

@inline function _Kcoulomb_norm(V,vc,gamma)
    r = _Kcoulomb(V,vc,gamma)
    r ./=sum(r)
end

"""
`ls_spectralext(Y,X,V,w,Nv::Int; normalization=:sum, normdim=:freq, λ = 1e-8, dims=3, coulomb = false, normalize=true)`

`Y` output\n
`X` sample locations\n
`V` scheduling signal\n
"""
function ls_spectralext(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer;  λ = 1e-8, coulomb = false, normalize=true)
    w       = w[:]
    N       = length(Y)
    Nf      = length(w)
    if coulomb # If Coulomb setting is activated, double the number of basis functions and clip the activation at zero velocity
        vc      = linspace(0,maximum(abs(V)),Nv+2)
        vc      = vc[2:end-1]
        vc      = [-vc[end:-1:1]; vc]
        Nv      = 2Nv
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K       = normalize ? (V,vc) -> _Kcoulomb_norm(V,vc,gamma) : (V,vc) -> _Kcoulomb(V,vc,gamma) # Use coulomb basis function instead
    else
        vc      = linspace(minimum(V),maximum(V),Nv)
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K       = normalize ? (V,vc) -> _K_norm(V,vc,gamma) : (V,vc) -> _K(V,vc,gamma)
    end

    M(w,X,V) = vec(vec(exp(im*w.*X))*K(V,vc)')'
    A        = zeros(Complex128,N,Nf*Nv)

    for n = 1:N
        A[n,:] = M(w,X[n],V[n])
    end

    params = real_complex_bs(A,Y,λ)
    real_params = [real(params); imag(params)]
    AA = [real(A) imag(A)]
    e = AA*real_params-Y
    Σ = var(e)*inv(AA'AA + λ*I)
    fva = 1-var(e)/var(Y)
    fva < 0.9 && warn("Fraction of variance explained = $(fva)")
    SpectralExt(Y, X, V, w, Nv, λ, coulomb, normalize, params, Σ)

end


# TODO: Behöver det fixas någon windowing i tid? Antagligen ja för riktiga signaler
