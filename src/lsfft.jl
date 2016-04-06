
""" `ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))`"""
function ls_spectral(y,t,f=(0:((length(y)-1)/2))/length(y))
    N = length(y)
    Nf = length(f)
    A = zeros(Complex128,N,Nf)
    for n = 1:N, fn=1:Nf
        A[n,fn] = exp(-2*pi*1im*f[fn]*t[n])
    end

    x = A\y
end

"""`ls_spectral_real(y,t,f=(0:((length(y)-1)/2))/length(y))`"""
function ls_spectral_real(y,t,f=(0:((length(y)-1)/2))/length(y))
    N = length(y)
    Nf = length(f)
    A = zeros(N,2Nf)
    for n = 1:N, fn=1:Nf
        phi = 2*pi*f[fn]*t[n]
        A[n,fn] = cos(phi)
        A[n,fn+Nf] = -sin(phi)

    end
    x = A\y
    # x = x[1:Nf] + 1im*x[Nf+1:end]
    print_with_color(:blue,"$(round(cond(A'A),2))\n")
    x = reshape(x,Nf,2)
end

"""`tls_spectral_real(y,t,f=(0:((length(y)-1)/2))/length(y))`"""
function tls_spectral_real(y,t,f=(0:((length(y)-1)/2))/length(y))
    N = length(y)
    Nf = length(f)
    A = zeros(N,2Nf)
    for n = 1:N, fn=1:Nf
        phi = 2*pi*f[fn]*t[n]
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
    x = reshape(x,Nf,2)
end

"""`ls_spectral(y,t,f,W::VecOrMat)`"""
function ls_spectral(y,t,f,W::VecOrMat)
    N = length(y)
    Nf = length(f)
    A = zeros(Complex128,N,Nf)
    for n = 1:N, fn=1:Nf
        A[n,fn] = exp(-2*pi*1im*f[fn]*t[n])
    end

    W = diagm(W)
    x   = (A'*W*A)\A'*W*y
end

"""`ls_spectral_real(y,t,f,W::VecOrMat)`"""
function ls_spectral_real(y,t,f,W::VecOrMat)
    N = length(y)
    Nf = length(f)
    A = zeros(N,2Nf)
    for n = 1:N, fn=1:Nf
        phi = 2*pi*f[fn]*t[n]
        A[n,fn] = cos(phi)
        A[n,fn+Nf] = -sin(phi)
    end

    W = diagm(W)
    x   = (A'*W*A)\A'*W*y
    print_with_color(:magenta,"$(round(cond(A'*W*A),2))\n")
    x = x[1:Nf] + 1im*x[Nf+1:end]
end




"""`lswindowpsd(y,t,freqs, nw = 10, noverlap = 0)`"""
function lswindowpsd(y,t,freqs, nw = 10, noverlap = 0)
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
        [0.5*(1 - cos(2*pi*k/(n-1))) for k=t]
    end
    W     = DSP.hanning(dpw)
    S       = 0.
    for j = 1:nw
        # W     = hanningjitter(t[inds])
        x     = ls_spectral_real(y[inds],t[inds],freqs,W)
        inds  = inds + (dpw - noverlap)
        # Power spectrum
        S = S + abs(x).^2
    end
    return S
end

"""`lswindowcsd(y,u,t,freqs, nw = 10, noverlap = 0)`"""
function lswindowcsd(y,u,t,freqs, nw = 10, noverlap = 0)
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
        S = S + xy.*conj(xu)
    end
    return S
end


# function lscohere(y,u,t,freqs, nw = 10, noverlap = 0)
#         Syu     = lswindowcsd(y,u,t,freqs, nw, noverlap)
#         Syy     = lswindowpsd(y,  t,freqs, nw, noverlap)
#         Suu     = lswindowpsd(u,  t,freqs, nw, noverlap)
#         Sch     = (abs(Syu).^2)./(Suu.*Syy);
# end

"""`lscohere(y,u,t,freqs, nw = 10, noverlap = 0)`"""
function lscohere(y,u,t,freqs, nw = 10, noverlap = 0)
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
        Syu = Syu + xy.*conj(xu)
        Syy = Syy + abs(xy).^2
        Suu = Suu + abs(xu).^2
    end
    Sch     = (abs(Syu).^2)./(Suu.*Syy);
    return Sch
end

_K(V,vc,gamma) = exp(-gamma*(V-vc).^2)

function _K_norm(V,vc,gamma)
    r = _K(V,vc,gamma)
    r ./=sum(r)
end

_Kcoulomb(V,vc,gamma) = _K(V,vc,gamma).*(sign(V) .== sign(vc))

function _Kcoulomb_norm(V,vc,gamma)
    r = _Kcoulomb(V,vc,gamma)
    r ./=sum(r)
end

"""``"""
function ls_spectralext(Y,X,V,w,Nv::Int; normalization=:sum, normdim=:freq, lambda = 1e-8, dims=3, coulomb = false, normalize=true, realaritm=true, kwargs...)
    w       = w[:]
    N       = length(Y)
    Nf      = length(w)
    if coulomb
        Nv      = 2Nv+1
        vc      = linspace(0,maximum(abs(V)),Nv+2)
        vc      = vc[2:end-1]
        vc      = [-vc[end:-1:1];0; vc]
        gamma   = 1/(abs(vc[1]-vc[2]))
        K(V,vc) = normalize ? _Kcoulomb_norm(V,vc,gamma) : _Kcoulomb(V,vc,gamma) # Use coulomb basis function instead
    else
        vc      = linspace(minimum(V),maximum(V),Nv)
        gamma   = 1/(abs(vc[1]-vc[2]))
        K(V,vc) = normalize ? _K_norm(V,vc,gamma) : _K(V,vc,gamma)
    end

    if realaritm
        M(w,X,V) = vec(vec([cos(w.*X)'; sin(w.*X)'])*K(V,vc)')'
        A        = zeros(N,2Nf*Nv)
    else
        M(w,X,V) = vec(vec(exp(im*w.*X)')*K(V,vc)')'
        A        = zeros(Complex128,N,Nf*Nv)
    end
    for n = 1:N
        A[n,:] = M(w,X[n],V[n])
    end

    params = ridgereg(A,Y,lambda, !realaritm)
    e = A*params-Y
    Σ = var(e)*inv(A'A + lambda*I)
    fva = 1-var(e)/var(Y)
    fva < 0.9 && warn("Fraction of variance explained = $(fva)")
    if realaritm
        x   = reshape(params,2,Nf,Nv)
        ax  = squeeze(sqrt(sum(x.^2,1)),1)
        px  = squeeze(atan2(x[2,:,:],x[1,:,:]),1)
    else
        x   = reshape(params,Nf,Nv)
        ax  = abs(x)
        px  = angle(x)
    end



    fg,vg = meshgrid(w,linspace(minimum(V),maximum(V),Nf == 100 ? 101 : 100)) # to guarantee that the broadcast below always works
    F = zeros(size(fg))
    P = zeros(size(fg))

    for j = 1:size(fg,1)
        for i = 1:size(vg,2) # freqs
            F[j,i] = (ax[j,:]*K(vg[j,i],vc))[1]
            P[j,i] = (px[j,:]*K(vg[j,i],vc))[1]
        end
    end



    nd = normdim == :freq ? 1 : 2
    normalizer = 1
    if normalization == :sum
        normalizer =   sum(F, nd)/size(F,nd)

    elseif normalization == :max
        normalizer =   maximum(F, nd)
    end
    F = F./normalizer

    if dims == 3
        fig = plot3d()
        for i = 1:Nf
            plot3d!(fg[i,:]'[:],vg[i,:]'[:],F[i,:]'[:]; kwargs...)
        end
        plot3d!(ylabel="\$v\$", xlabel="\$ω\$")#, zlabel="\$f(v)\$")
    else
        figF = plot()
        figP = plot()
        for i = 1:Nf
            plot!(figF,vg[i,:]'[:],F[i,:]'[:]; lab="\$ω = $(round(fg[i,1],1))\$", kwargs...)
            plot!(figP,vg[i,:]'[:],P[i,:]'[:]; lab="\$ω = $(round(fg[i,1],1))\$", kwargs...)
        end
        plot!(figF,xlabel="\$v\$", ylabel="\$A(v)\$", title="Estimated functional dependece \$A(v)\$\n Normalization: $normalization, along dim $normdim, "*(realaritm ? "real" : "complex")*" arithmetics")#, zlabel="\$f(v)\$")

        plot!(figP,xlabel="\$v\$", ylabel="\$ϕ(v)\$", title="Estimated functional dependece \$ϕ(v)\$\n Normalization: $normalization, along dim $normdim, "*(realaritm ? "real" : "complex")*" arithmetics")#, zlabel="\$f(v)\$")

    end

    figF
end



# TODO: Behöver det fixas någon windowing i tid? Antagligen ja för riktiga signaler
