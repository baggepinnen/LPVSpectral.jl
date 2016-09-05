function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end

@userplot Periodogram

@recipe function plot_periodogram(p::Periodogram; Fs=1)
    seriestype := :spectrum
    title --> "Periodogram"
    # yscale --> :log10
    # xguide --> "Frequency / [\$F_s\$]"
    if length(p.args) > 1
        x = p.args[1]
        y = p.args[2]
    else
        y = p.args[1]
        x = linspace(-Fs/2,Fs/2,length(y))
    end
    delete!(d,:Fs)
    y = abs(fft(y))
    (x,y)
end


@recipe function plot_spectrum(::Type{Val{:spectrum}}, plt::Plot)
    title --> "Spectrum"
    yscale --> :log10
    xguide --> "Frequency / [\$F_s\$]"
    seriestype := :path
end


# @userplot SchedFunc

@recipe function plot_schedfunc(se::SpectralExt; normalization=:none, normdim=:freq, dims=3, bounds=true, nMC = 5_000, phase = false, mcmean=false)
    xi,V,X,w,Nv,coulomb,normalize = se.x,se.V,se.X,se.w,se.Nv,se.coulomb,se.normalize
    Nf = length(w)
    x = reshape_params(xi,Nf)
    ax  = abs(x)
    px  = angle(x)
    if coulomb
        vc      = linspace(0,maximum(abs(V)),Nv/2+2)
        vc      = vc[2:end-1]
        vc      = [-vc[end:-1:1]; vc]
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K = normalize ? V->_Kcoulomb_norm(V,vc,gamma) : V->_Kcoulomb(V,vc,gamma) # Use coulomb basis function instead
    else
        vc      = linspace(minimum(V),maximum(V),Nv)
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K = normalize ? V->_K_norm(V,vc,gamma) : V->_K(V,vc,gamma)
    end


    fg,vg = meshgrid(w,linspace(minimum(V),maximum(V),Nf == 100 ? 101 : 100)) # to guarantee that the broadcast below always works
    F = zeros(size(fg))
    FB = zeros(size(fg)...,nMC)
    P = zeros(size(fg))
    PB = zeros(size(fg)...,nMC)
    if bounds
        cn = ComplexNormal(se.x,se.Σ)
        zi = rand(cn,nMC) # Draw several random parameters from the posterior distribution
    end

    for j = 1:size(fg,1)
        for i = 1:size(vg,2)
            ϕ = K(vg[j,i]) # Kernel activation vector
            F[j,i] = abs(vecdot(x[j,:],ϕ))
            P[j,i] = angle(vecdot(x[j,:],ϕ))
            if bounds
                for iMC = 1:nMC
                    zii = zi[iMC,j:Nf:end][:]
                    FB[j,i,iMC] = abs(vecdot(zii,ϕ))
                    if phase
                        PB[j,i,iMC] = angle(vecdot(zii,ϕ))
                    end
                end
            end
        end
    end
    FB = sort(FB,3)
    lim = 1000
    FBl = FB[:,:,nMC ÷ lim]
    FBu = FB[:,:,nMC - (nMC ÷ lim)]
    FBm = squeeze(mean(FB,3),3)
    PB = sort(PB,3)
    PBl = PB[:,:,nMC ÷ lim]
    PBu = PB[:,:,nMC - (nMC ÷ lim)]
    PBm = squeeze(mean(PB,3),3)

    nd = normdim == :freq ? 1 : 2
    normalizer = 1.
    if normalization == :sum
        normalizer =   sum(F, nd)/size(F,nd)
    elseif normalization == :max
        normalizer =   maximum(F, nd)
    end
    F = F./normalizer
    delete!(d, :normalization)
    delete!(d, :normdim)

    if dims == 3
        delete!(d, :dims)
        yguide --> "\$v\$"
        xguide --> "\$\\omega\$"
        # zguide := "\$f(v)\$"
        for i = 1:Nf
            @series begin
                seriestype := path3d
                (fg[i,:]'[:],vg[i,:]'[:],F[i,:]'[:])
            end
        end
    else

        for i = 1:Nf
            xguide --> "\$v\$"
            yguide --> "\$A(v)\$"
            title --> "Estimated functional dependece \$A(v)\$\n"# Normalization: $normalization, along dim $normdim")#, zlabel="\$f(v)\$")
            @series begin
                label --> "\$\\omega = $(round(fg[i,1]/pi,1))\\pi\$"
                m = mcmean ? FBm[i,:]'[:] : F[i,:]'[:]
                if bounds
                    # fillrange := FBu[i,:]'[:]
                    ribbon := (-FBl[i,:]'[:] + m, FBu[i,:]'[:] - m)
                end
                (vg[i,:]'[:],m)
            end
        end
        if phase
            for i = 1:Nf
                xguide --> "\$v\$"
                yguide --> "\$\\phi(v)\$"
                linestyle := :dashdot
                @series begin
                    label --> "\$\\phi\$"
                    fillalpha := 0.1
                    pi = P[i,:]'[:]
                    if bounds
                        ribbon := (-PBl[i,:]'[:] + pi, PBu[i,:]'[:] - pi)
                    end
                    (vg[i,:]'[:],pi)
                end
            end
        end

    end
    delete!(d, :phase)
    delete!(d, :bounds)
    delete!(d, :nMC)
    delete!(d, :mcmean)

    nothing

end


@recipe function plot_spectralext(::Type{Val{:spectralext}}, x, y, z)
    xi,V,w,Nv,coulomb,normalize = y.x,y.V,y.w,y.Nv,y.coulomb,y.normalize
    title --> "Spectrum"
    Nf = length(w)
    x = reshape_params(xi,Nf)
    ax  = abs2(x)
    px  = angle(x)
    ax
end
