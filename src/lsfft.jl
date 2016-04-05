
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


# """``"""
# function ls_spectralext(y::Vector,t::Vector,v::Vector,f,Nv; plotres = false)
#     f       = f[:]
#     N       = length(y)
#     Nf      = length(f)
#     A       = zeros(N,2Nf*(2Nv+1))
#
#
#     vc      = linspace(0,max(abs(v)),Nv+2)
#     vc      = vc[2:end-1]
#     vc      = [-vc[end:-1:1];0; vc]
#     gamma   = 1/(abs(vc[1]-vc[2]))
#
#     function K(v,vc)
#         r = exp(-gamma*(v-vc).^2).*(sign(v) .== sign(vc))
#         r ./=sum(r)
#     end
#
#
#     M(f,t,v) = vec(vec([cos(f.*t)'; sin(f.*t)'])*K(v,vc)')'
#
#     for n = 1:N
#         A[n,:] = M(f,t[n],v[n])
#     end
#
#     lambda = 1e-3;
#     w   = (A'*A + lambda*eye(2Nf*(2Nv+1)))\(A'*y)
#     x   = reshape(w,2,Nf,(2Nv+1));
#     ax  = squeeze(sqrt(sum(x.^2,1)));
#     px  = squeeze(atan2(x[2,:,:],x[1,:,:]));
#
#
#     fva = 1-var(A*w-y)/var(y)
#
#
#     [fg,vg] = meshgrid(f,linspace(minimum(v),maximum(v),20))
#     F = zeros(size(fg))
#     P = zeros(size(fg))
#
#     for j = 1:size(fg,1)
#         for i = 1:size(vg,2) # freqs
#             F[j,i] = ax[i,:]*K(vg[j,i],vc)
#             P[j,i] = px[i,:]*K(vg[j,i],vc)
#         end
#     end
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
#         if false
#             figure,
#             contourf(fg',vg',(F)',linspace(min(v),max(v),100),':')
#             xlabel("Frequency")
#             ylabel("Velocity")
#             colormap("jet")
#             colorbar
#         end
#         if true
#             figure,
#             subplot(1,2,1)
#             imagesc(fg(1,:)',vg(:,1),F)
#             xlabel("Frequency")
#             ylabel("Velocity")
#             colormap("jet")
#             colorbar
#             subplot(1,2,2)
#             plot(fg(1,:)',mag2db(mean(F,1)),'o')
#             xlabel("Frequency")
#             ylabel("log(Power) [dB]")
#             grid on
#             xlim([0 fg(1,end)])
#         end
#     end
#
#     # % subplot(212), waterfall(fg',vg',P')
#     # % zlabel('Phase')
#     # % ylabel('Velocity')
#     # % alpha(0.6)
#     # % xlabel('Frequency')
#     # % figure, imagesc(F)
#     # % figure, imagesc(P)
#
# end
