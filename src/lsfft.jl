import Base: start, done, next


"""`ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y); λ=0)`

perform spectral estimation using the least-squares method
`y` is the signal to be analyzed
`t` is the sampling points
`f` is a vector of frequencies
"""
function ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y); λ=0)
    N = length(y)
    Nf = length(f)
    A = [exp(2π*f[fn]*t[n]) for n = 1:N, fn = 1:Nf]
    x = real_complex_bs(A,b,λ)
    info("Condition number: $(round(cond(A'A),2))\n")
end


"""`ls_spectral(y,t,f,W::AbstractVector)`
`W` is a vector of weights, for weighted least-squares
"""
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
    info("Condition number: $(round(cond(A'*W*A),2))\n")
    x = complex(x[1:Nf], x[Nf+1:end])
end


"""`tls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))`
Perform total least-squares spectral estimation using the SVD-method. See `ls_spectral` for additional help
"""
function tls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))
    N = length(y)
    Nf = length(f)
    A = zeros(N,2Nf)
    for n = 1:N, fn=1:Nf
        phi = 2π*f[fn]*t[n]
        A[n,fn] = cos(phi)
        A[n,fn+Nf] = -sin(phi)

    end
    AA    = [A y]
    U,S,V = svd(AA)
    m,n,p = N,2Nf,1
    V21   = V[1:n,n+1:end]
    V22   = V[n+1:end,n+1:end]
    x     = -V21/V22
    # x = x[1:Nf] + 1im*x[Nf+1:end]
    info("Condition number: $(round(cond(AA'AA),2))\n")
    x = complex(x[1:Nf], x[Nf+1:end])
end


"""`ls_windowpsd(y,t,freqs, nw = 10, noverlap = -1, window_func=rect)`

perform widowed spectral estimation using the least-squares method.
`window_func` defaults to `Windows.rect`

See `ls_spectral` for additional help.
"""
function ls_windowpsd(y,t,freqs, nw = 10, noverlap = -1, window_func=rect)
    windows = Windows2(y,t,nw,noverlap,window_func)
    S       = zeros(length(freqs))
    for (y,t) in windows
        x  = ls_spectral(y,t,freqs,windows.W)
        S += abs2(x)
    end
    return S
end

"""`ls_windowcsd(y,u,t,freqs, nw = 10, noverlap = -1, window_func=rect)`

Perform windowed cross spectral density estimation using the least-squares method.

`y` and `u` are the two signals to be analyzed and `t::AbstractVector` are their sampling points
`window_func` defaults to `Windows.rect`

See `ls_spectral` for additional help.
"""
function ls_windowcsd(y,u,t,freqs, nw = 10, noverlap = -1, window_func=rect)
    S       = zeros(Complex128,length(freqs))
    windows = Windows2(y,t,nw,noverlap,window_func)
    for (y,t) in windows
        xy      = ls_spectral(y,t,freqs,windows.W)
        xu      = ls_spectral(u,t,freqs,windows.W)
        # Cross spectrum
        S += xy.*conj(xu)
    end
    return S
end




# function lscohere(y,u,t,freqs, nw = 10, noverlap = -1)
#         Syu     = lswindowcsd(y,u,t,freqs, nw, noverlap)
#         Syy     = lswindowpsd(y,  t,freqs, nw, noverlap)
#         Suu     = lswindowpsd(u,  t,freqs, nw, noverlap)
#         Sch     = (abs(Syu).^2)./(Suu.*Syy);
# end

"""`ls_cohere(y,u,t,freqs, nw = 10, noverlap = -1)`

Perform spectral coherence estimation using the least-squares method.
See also `ls_windowcsd` and `ls_spectral` for additional help.
"""
function ls_cohere(y,u,t,freqs, nw = 10, noverlap = -1)
    Syy       = zeros(length(freqs))
    Suu       = zeros(length(freqs))
    Syu       = zeros(Complex128,length(freqs))
    windows   = Windows3(y,t,u,nw,noverlap,hanning)
    for (y,t,u) in windows
        xy      = ls_spectral(y,t,freqs,windows.W)
        xu      = ls_spectral(u,t,freqs,windows.W)
        inds   += (dpw - noverlap)
        # Cross spectrum
        Syu += xy.*conj(xu)
        Syy += abs2(xy)
        Suu += abs2(xu)
    end
    Sch     = abs2(Syu)./(Suu.*Syy)
    return Sch
end

@inline _K(V,vc,gamma) = exp(-gamma*(V.-vc).^2)

@inline function _K_norm(V,vc,gamma)
    r = _K(V,vc,gamma)
    r ./=sum(r)
end

@inline _Kcoulomb(V,vc,gamma) = _K(V,vc,gamma).*(sign(V) .== sign(vc))

@inline function _Kcoulomb_norm(V,vc,gamma)
    r = _Kcoulomb(V,vc,gamma)
    r ./=sum(r)
end

"""psd(se::SpectralExt)
Compute the power spectral density for a SpectralExt object

See also `ls_windowpsd_lpv`
"""
function psd(se::SpectralExt)
    rp = LPVSpectral.reshape_params(copy(se.x),length(se.w))
    return sum(rp,2) |> abs2
end

"""
`ls_spectral_lpv(Y,X,V,w,Nv::Int; λ = 1e-8, coulomb = false, normalize=true)`

Perform LPV spectral estimation using the method presented in
Bagge Carlson et al. "Linear Parameter-Varying Spectral Decomposition."
See the paper for additional details.

`Y` output\n
`X` sample locations\n
`V` scheduling signal\n
`w` frequency vector\n
`Nv` number of basis functions\n
`λ` Regularization parameter\n
`coulomb` Assume discontinuity at `v=0` (useful for signals where, e.g., Coulomb friction might cause issues.)\n
`normalize` Use normalized basis functions (See paper for details).

The method will issue a warning if less than 90% of the variance in `Y` is described by the estimated model. If this is the case, try increasing either the number of frequencies or the number of basis functions per frequency. Alternatively, try lowering the regularization parameter `λ`.

See also `psd` and `ls_windowpsd_lpv`
"""
function ls_spectral_lpv(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer;  λ = 1e-8, coulomb = false, normalize=true)
    w        = w[:]
    N        = length(Y)
    Nf       = length(w)
    K        = basis_activation_func(V,Nv,normalize,coulomb)
    M(w,X,V) = vec(vec(exp(im*w.*X))*K(V)')'
    A        = zeros(Complex128,N,Nf*Nv*(coulomb ? 2 : 1))

    for n = 1:N
        A[n,:] = M(w,X[n],V[n])
    end

    params      = real_complex_bs(A,Y,λ)
    real_params = [real(params); imag(params)]
    AA          = [real(A) imag(A)]
    e           = AA*real_params-Y
    Σ           = var(e)*inv(AA'AA + λ*I)
    fva         = 1-var(e)/var(Y)
    fva < 0.9 && warn("Fraction of variance explained = $(fva)")
    SpectralExt(Y, X, V, w, Nv, λ, coulomb, normalize, params, Σ)

end

"""ls_windowpsd_lpv(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer, nw::Int=10, noverlap=0;  kwargs...)

Perform windowed psd estimation using the LPV method. A rectangular window is always used.

See `?ls_spectral_lpv` for additional help.
"""
function ls_windowpsd_lpv(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer, nw::Int=10, noverlap=0;  kwargs...)
    S       = zeros(length(w))
    windows = Windows3(Y,X,V,nw,noverlap,rect) # ones produces a rectangular window
    for (y,x,v) in windows
        x  = ls_spectral_lpv(y,x,v,w,Nv; kwargs...)
        rp = reshape_params(x.x,length(w))
        S += sum(rp,2) |> abs2
    end
    return S

end
