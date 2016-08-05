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

@recipe function plot_schedfunc(se::SpectralExt; normalization=:none, normdim=:freq, dims=3, bounds=true, nMC = 5_000)
    xi,V,w,Nv,coulomb,normalize = se.x,se.V,se.w,se.Nv,se.coulomb,se.normalize
    Nf = length(w)
    x = reshape_params(xi,Nf)
    ax  = abs(x)
    px  = angle(x)
    if coulomb
        Nv      = 2Nv+1
        vc      = linspace(0,maximum(abs(V)),Nv+2)
        vc      = vc[2:end-1]
        vc      = [-vc[end:-1:1];0; vc]
        gamma   = 1/(abs(vc[1]-vc[2]))
        K = normalize ? V->_Kcoulomb_norm(V,vc,gamma) : V->_Kcoulomb(V,vc,gamma) # Use coulomb basis function instead
    else
        vc      = linspace(minimum(V),maximum(V),Nv)
        gamma   = 1/(abs(vc[1]-vc[2]))
        K = normalize ? V->_K_norm(V,vc,gamma) : V->_K(V,vc,gamma)
    end


    fg,vg = meshgrid(w,linspace(minimum(V),maximum(V),Nf == 100 ? 101 : 100)) # to guarantee that the broadcast below always works
    F = zeros(size(fg))
    FB = zeros(size(fg)...,nMC)
    P = zeros(size(fg))
    if bounds
        cn = ComplexNormal(se.x,se.Σ)
        zi = rand(cn,nMC)
        az  = abs(zi)
        pz  = angle(zi)
    end

    for j = 1:size(fg,1)
        for i = 1:size(vg,2) # freqs
            r = K(vg[j,i])
            F[j,i] = vecdot(ax[j,:],r)
            P[j,i] = vecdot(px[j,:],r)
            if bounds
                for iMC = 1:nMC
                    azi = az[iMC,j:Nf:end][:]
                    FB[j,i,iMC] = vecdot(azi,r)
                end
            end
        end
    end
    FB = sort(FB,3)
    FBl = FB[:,:,nMC ÷ 20]
    FBu = FB[:,:,nMC - (nMC ÷ 20)]

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
                label --> "\$\\omega = $(round(fg[i,1],1))\$"
                if bounds
                    ribbon := (FBl[i,:]'[:] - F[i,:]'[:], FBu[i,:]'[:] - F[i,:]'[:])
                end
                (vg[i,:]'[:],F[i,:]'[:])
            end
        end

    end
    delete!(d, :bounds)
    delete!(d, :nMC)

    nothing

end


@recipe function plot_spectralext(::Type{Val{:spectralext}}, x, y, z)
    xi,V,w,Nv,coulomb,normalize = y.x,y.V,y.w,y.Nv,y.coulomb,y.normalize
    title --> "Spectrum"
    Nf = length(w)
    x = reshape_params(xi,Nf)
    ax  = abs(x)
    px  = angle(x)
    ax
end
