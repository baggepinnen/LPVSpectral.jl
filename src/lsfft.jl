import Base: start, done, next

immutable Windows2
    y::AbstractVector
    t::AbstractVector
    nw::Int
    dpw::Int
    noverlap::Int
    W
    function Windows2(y,t,nw,noverlap,window_func)
        N       = length(y)
        dpw     = floor(Int64,N/nw)
        if noverlap == 0
            noverlap = round(Int64,dpw/2)
        end
        W       = window_func(dpw)
        Windows2(y,t,nw,dpw,noverlap,W)
    end
end


Base.start(::Windows2) = 1
function Base.next(w::Windows2, state)
    inds =  (1:w.dpw)+(state-1)*(w.dpw-w.noverlap)
    ((w.y[inds],w.t[inds]), state+1)
end
Base.done(w::Windows2, state) = state > w.nw;
Base.length(w::Windows2) = w.nw;

immutable Windows3
    y::AbstractVector
    t::AbstractVector
    v::AbstractVector
    nw::Int
    dpw::Int
    noverlap::Int
    W
end

function Windows3(y,t,v,nw::Integer,noverlap::Integer,window_func::Function)
    N       = length(y)
    dpw     = floor(Int64,N/nw)
    if noverlap == 0
        noverlap = round(Int64,dpw/2)
    end
    W       = window_func(dpw)
    Windows3(y,t,v,nw,dpw,noverlap,W)
end


Base.start(::Windows3) = 1
function Base.next(w::Windows3, state)
    inds =  (1:w.dpw)+(state-1)*(w.dpw-w.noverlap)
    ((w.y[inds],w.t[inds],w.v[inds]), state+1)
end
Base.done(w::Windows3, state) = state > w.nw;
Base.length(w::Windows3) = w.nw;


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
    print_with_color(:magenta,"$(round(cond(A'A),2))\n")
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
    print_with_color(:magenta,"$(round(cond(A'*W*A),2))\n")
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


"""`ls_windowpsd(y,t,freqs, nw = 10, noverlap = 0)`

perform widowed spectral estimation using the least-squares method. See `ls_spectral` for additional help.
"""
function ls_windowpsd(y,t,freqs, nw = 10, noverlap = 0)
    # function hanningjitter(t)
    #     n = length(t)
    #     t = t-t[1]
    #     t *= (n-1)/t[end]
    #     [0.5*(1 - cos(2π*k/(n-1))) for k=t]
    # end
    windows = Windows2(y,t,nw,noverlap,hanning)
    S       = 0.
    for (y,t) in windows
        # W     = hanningjitter(t[inds])
        x     = ls_spectral(y,t,freqs,windows.W)
        # Power spectrum
        S += abs2(x)
    end
    return S
end

"""`ls_windowcsd(y,u,t,freqs, nw = 10, noverlap = 0)`

Perform windowed cross spectral density estimation using the least-squares method.

`y` and `u` are the two signals to be analyzed and `t::AbstractVector` are their sampling points
See `ls_spectral` for additional help.
"""
function ls_windowcsd(y,u,t,freqs, nw = 10, noverlap = 0)
    S       = 0.
    windows = Windows2(y,t,nw,noverlap,hanning)
    for (y,t) in windows
        xy      = ls_spectral(y,t,freqs,windows.W)
        xu      = ls_spectral(u,t,freqs,windows.W)
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

"""`ls_cohere(y,u,t,freqs, nw = 10, noverlap = 0)`

Perform spectral coherence estimation using the least-squares method.
See also `ls_windowcsd` and `ls_spectral` for additional help.
"""
function ls_cohere(y,u,t,freqs, nw = 10, noverlap = 0)
    Syy       = 0.
    Suu       = 0.
    Syu       = zero(Complex128)
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

"""
`ls_spectralext(Y,X,V,w,Nv::Int; λ = 1e-8, coulomb = false, normalize=true)`

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
"""
function ls_spectralext(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer;  λ = 1e-8, coulomb = false, normalize=true)
    w        = w[:]
    N        = length(Y)
    Nf       = length(w)
    K        = basis_activation_func(V,Nv,normalize,coulomb)
    M(w,X,V) = vec(vec(exp(im*w.*X))*K(V)')'
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

"""ls_windowpsd_ext(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer, nw::Int=10, noverlap=0;  kwargs...)

Perform windowed psd estimation using the LPV method.
"""
function ls_windowpsd_ext(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer, nw::Int=10, noverlap=0;  kwargs...)
    S       = 0.
    windows = Windows3(Y,X,V,nw,noverlap,ones) # ones produces a rectangular window
    for (y,x,v) in windows
        x  = ls_spectralext(y,x,v,w,Nv; kwargs...)
        rp = reshape_params(x.x,length(w))
        S += sum(rp,2) |> abs2
    end
    return S

end
