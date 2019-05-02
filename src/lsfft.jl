default_freqs(t, nw=1) = LinRange(0,0.5-1/length(t)/2,length(t)÷2)

function check_freq(f)
    zerofreq = findfirst(iszero, f)
    zerofreq !== nothing && zerofreq != 1 && throw(ArgumentError("If zero frequency is included it must be the first frequency"))
    zerofreq
end

function get_fourier_regressor(t,f)
    zerofreq = check_freq(f)
    N  = length(t)
    Nf = length(f)
    Nreg = zerofreq === nothing ? 2Nf : 2Nf-1
    # N >= Nreg || throw(ArgumentError("Too many frequency components $Nreg > $N"))
    A  = zeros(N,Nreg)
    sinoffset = Nf
    for fn=1:Nf
        if fn == zerofreq
            sinoffset = Nf-1
        end
        for n = 1:N
            phi        = 2π*f[fn]*t[n]
            A[n,fn]    = cos(phi)
            if fn != zerofreq
                A[n,fn+sinoffset] = -sin(phi)
            end
        end
    end
    A, zerofreq
end

"""`x,f = ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y); λ=0)`

perform spectral estimation using the least-squares method
`y` is the signal to be analyzed
`t` is the sampling points
`f` is a vector of frequencies

See also `ls_sparse_spectral` `tls_spectral`
"""
function ls_spectral(y,t,f=default_freqs(t); λ=1e-10)
    A, zerofreq = get_fourier_regressor(t,f)
    x = fourier_solve(A,y,zerofreq,λ)
    # @info("Condition number: $(round(cond(A'A), digits=2))\n")
    x, f
end



"""`x,f = ls_spectral(y,t,f,W::AbstractVector)`
`W` is a vector of weights, same length as `y`, for weighted least-squares
"""
function ls_spectral(y,t,f,W::AbstractVector)
    A, zerofreq = get_fourier_regressor(t,f)
    Wd = Diagonal(W)
    x = (A'Wd*A + 1e-10I)\(A'Wd)*y
    # @info("Condition number: $(round(cond(A'*Wd*A), digits=2))\n")
    fourier2complex(x,zerofreq), f
end



"""`x,f = tls_spectral(y,t,f=(0:((length(y)-1)))/length(y))`
Perform total least-squares spectral estimation using the SVD-method. See `ls_spectral` for additional help
"""
function tls_spectral(y,t,f=default_freqs(t))
    zerofreq = check_freq(f)
    A, zerofreq = get_fourier_regressor(t,f)
    AA    = [A y]
    # s     = svd(AA, full=true)
    _,_,Vt = LAPACK.gesvd!('S','S',AA)
    n     = size(A,2)
    V21   = Vt[n+1,1:n]
    V22   = Vt[n+1,n+1]
    x     = -V21/V22

    fourier2complex(x,zerofreq), f
end


"""`S,f = ls_windowpsd(y,t,freqs; nw = 10, noverlap = -1, window_func=rect, estimator=ls_spectral, kwargs...)`

perform widowed spectral estimation using the least-squares method.
`window_func` defaults to `Windows.rect`

`estimator` is the spectral estimatio function to use, default is `ls_spectral`. For sparse estimation, try
`estimator = ls_sparse_spectral` See `ls_sparse_spectral` for more help. `kwargs` are passed to `estimator`.

See `ls_spectral` for additional help.
"""
function ls_windowpsd(y,t,freqs=nothing; nw = 10, noverlap = -1, window_func=rect, estimator=ls_spectral, kwargs...)
    freqs === nothing && (freqs = default_freqs(t,nw))
    windows = Windows2(y,t,nw,noverlap,window_func)
    S       = zeros(length(freqs))
    for (yi,ti) in windows
        x  = estimator(yi,ti,freqs,windows.W; kwargs...)[1]
        S += nw .* abs2.(x)
    end
    return S, freqs
end

"""`ls_windowcsd(y,u,t,freqs; nw = 10, noverlap = -1, window_func=rect, estimator=ls_spectral, kwargs...)`

Perform windowed cross spectral density estimation using the least-squares method.

`y` and `u` are the two signals to be analyzed and `t::AbstractVector` are their sampling points
`window_func` defaults to `Windows.rect`

`estimator` is the spectral estimatio function to use, default is `ls_spectral`. For sparse estimation, try
`estimator = ls_sparse_spectral` See `ls_sparse_spectral` for more help.

See `ls_spectral` for additional help.
"""
function ls_windowcsd(y,u,t,freqs=nothing; nw = 10, noverlap = -1, window_func=rect, estimator=ls_spectral, kwargs...)
    freqs === nothing && (freqs = default_freqs(t,nw))
    S       = zeros(ComplexF64,length(freqs))
    windowsy = Windows2(y,t,nw,noverlap,window_func)
    windowsu = Windows2(u,t,nw,noverlap,window_func)
    for ((y,t), (u,_)) in zip(windowsy, windowsu)
        xy = estimator(y,t,freqs,windowsy.W; kwargs...)[1]
        xu = estimator(u,t,freqs,windowsu.W; kwargs...)[1]
        # Cross spectrum
        S += nw.*xy.*conj.(xu)
    end
    return S, freqs
end




# function lscohere(y,u,t,freqs, nw = 10, noverlap = -1)
#         Syu     = lswindowcsd(y,u,t,freqs, nw, noverlap)
#         Syy     = lswindowpsd(y,  t,freqs, nw, noverlap)
#         Suu     = lswindowpsd(u,  t,freqs, nw, noverlap)
#         Sch     = (abs(Syu).^2)./(Suu.*Syy);
# end

"""`ls_cohere(y,u,t,freqs; nw = 10, noverlap = -1, estimator=ls_spectral, kwargs...)`

Perform spectral coherence estimation using the least-squares method.

`estimator` is the spectral estimatio function to use, default is `ls_spectral`. For sparse estimation, try
`estimator = ls_sparse_spectral` See `ls_sparse_spectral` for more help.
See also `ls_windowcsd` and `ls_spectral` for additional help.
"""
function ls_cohere(y,u,t,freqs=nothing; nw = 10, noverlap = -1, estimator=ls_spectral, kwargs...)
    freqs === nothing && (freqs = default_freqs(t,nw))
    Syy     = zeros(length(freqs))
    Suu     = zeros(length(freqs))
    Syu     = zeros(ComplexF64,length(freqs))
    windows = Windows3(y,t,u,nw,noverlap,hanning)
    for (y,t,u) in windows
        xy      = estimator(y,t,freqs,windows.W; kwargs...)[1]
        xu      = estimator(u,t,freqs,windows.W; kwargs...)[1]
        # Cross spectrum
        Syu .+= xy.*conj.(xu)
        Syy .+= abs2.(xy)
        Suu .+= abs2.(xu)
    end
    Sch = abs2.(Syu)./(Suu.*Syy)
    return Sch, freqs
end

@inline _K(V,vc,gamma) = exp.(-gamma*(V.-vc).^2)

@inline function _K_norm(V,vc,gamma)
    r = _K(V,vc,gamma)
    r ./=sum(r)
end

@inline _Kcoulomb(V,vc,gamma) = _K(V,vc,gamma).*(sign.(V) .== sign.(vc))

@inline function _Kcoulomb_norm(V,vc,gamma)
    r = _Kcoulomb(V,vc,gamma)
    r ./=sum(r)
end

"""psd(se::SpectralExt)
Compute the power spectral density For a SpectralExt object

See also `ls_windowpsd_lpv`
"""
function psd(se::SpectralExt)
    rp = LPVSpectral.reshape_params(copy(se.x),length(se.w))
    return abs2.(sum(rp,dims=2))
end

"""
`ls_spectral_lpv(Y,X,V,w,Nv::Int; λ = 1e-8, coulomb = false, normalize=true)`

Perform LPV spectral estimation using the method presented in
Bagge Carlson et al. "Linear Parameter-Varying Spectral Decomposition."
See the paper For additional details.

`Y` output\n
`X` sample locations\n
`V` scheduling signal\n
`w` frequency vector\n
`Nv` number of basis functions\n
`λ` Regularization parameter\n
`coulomb` Assume discontinuity at `v=0` (useful for signals where, e.g., Coulomb friction might cause issues.)\n
`normalize` Use normalized basis functions (See paper for details).

The method will issue a warning If less than 90% of the variance in `Y` is described by the estimated model. If this is the case, try increasing either the number of frequencies or the number of basis functions per frequency. Alternatively, try lowering the regularization parameter `λ`.

See also `psd`, `ls_sparse_spectral_lpv` and `ls_windowpsd_lpv`
"""
function ls_spectral_lpv(Y::AbstractVector,X::AbstractVector,V::AbstractVector,w,Nv::Integer;  λ = 1e-8, coulomb = false, normalize=true)
    w        = w[:]
    N        = length(Y)
    Nf       = length(w)
    K        = basis_activation_func(V,Nv,normalize,coulomb)
    M(w,X,V) = vec(vec(exp.(im*w.*X))*K(V)')'
    A        = zeros(ComplexF64,N, ifelse(coulomb,2,1)*Nf*Nv)
    for n = 1:N
        A[n,:] = M(w,X[n],V[n])
    end

    params      = real_complex_bs(A,Y,λ)
    real_params = [real.(params); imag.(params)]
    AA          = [real.(A) imag.(A)]
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
        S += abs2.(sum(rp,dims=2))
    end
    return S

end
