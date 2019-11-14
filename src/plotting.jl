function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end


@recipe function plot_periodogram(p::DSP.Periodograms.TFR)
    seriestype := :spectrum
    title --> "Periodogram"
    p.freq, p.power
end


@recipe function plot_spectrum(::Type{Val{:spectrum}}, plt::AbstractPlot)
    title --> "Spectrum"
    yscale --> :log10
    xguide --> "Frequency"
    seriestype := :path
end


# @userplot SchedFunc

@recipe function plot_schedfunc(se::SpectralExt; normalization=:none, normdim=:freq, dims=2, bounds=true, nMC = 5_000, phase = false, mcmean=false)
    xi,V,X,w,Nv,coulomb,normalize = se.x,se.V,se.X,se.w,se.Nv,se.coulomb,se.normalize
    Nf = length(w)
    x  = reshape_params(xi,Nf)
    ax = abs.(x)
    px = angle.(x)
    K  = basis_activation_func(V,Nv,normalize,coulomb)

    fg,vg = meshgrid(w,LinRange(minimum(V),maximum(V),Nf == 100 ? 101 : 100)) # to guarantee that the broadcast below always works
    F  = zeros(size(fg))
    FB = zeros(size(fg)...,nMC)
    P  = zeros(size(fg))
    PB = zeros(size(fg)...,nMC)
    bounds = bounds && se.Σ != nothing
    if bounds
        cn = ComplexNormal(se.x,se.Σ)
        zi = LPVSpectral.rand(cn,nMC) # Draw several random parameters from the posterior distribution
    end

    for j = 1:size(fg,1)
        for i = 1:size(vg,2)
            ϕ = K(vg[j,i]) # Kernel activation vector
            F[j,i] = abs(dot(x[j,:],ϕ))
            P[j,i] = angle(dot(x[j,:],ϕ))
            if bounds
                for iMC = 1:nMC
                    zii = zi[iMC,j:Nf:end][:]
                    FB[j,i,iMC] = abs(dot(zii,ϕ))
                    if phase
                        PB[j,i,iMC] = angle(dot(zii,ϕ))
                    end
                end
            end
        end
    end
    FB = sort(FB,dims=3)
    lim = 10
    FBl = FB[:,:,nMC ÷ lim]
    FBu = FB[:,:,nMC - (nMC ÷ lim)]
    FBm = dropdims(mean(FB,dims=3),dims=3)
    PB = sort(PB,dims=3)
    PBl = PB[:,:,nMC ÷ lim]
    PBu = PB[:,:,nMC - (nMC ÷ lim)]
    PBm = dropdims(mean(PB,dims=3),dims=3)

    nd = normdim == :freq ? 1 : 2
    normalizer = 1.
    if normalization == :sum
        normalizer =   sum(F, dims=nd)/size(F,nd)
    elseif normalization == :max
        normalizer =   maximum(F, dims=nd)
    end
    F = F./normalizer
    delete!(plotattributes, :normalization)
    delete!(plotattributes, :normdim)

    if dims == 3
        delete!(plotattributes, :dims)
        yguide --> "\$v\$"
        xguide --> "\$\\omega\$"
        # zguide := "\$f(v)\$"
        for i = 1:Nf
            @series begin
                seriestype := path3d
                fg[i,:],vg[i,:],F[i,:]
            end
        end
    else

        for i = 1:Nf
            xguide --> "\$v\$"
            yguide --> "\$A(v)\$"
            title --> "Estimated functional dependece \$A(v)\$\n"# Normalization: $normalization, along dim $normdim")#, zlabel="\$f(v)\$")
            @series begin
                label --> "\$\\omega = $(round(fg[i,1]/pi,sigdigits=1))\\pi\$"
                m = mcmean && bounds ? FBm[i,:] : F[i,:]
                if bounds
                    # fillrange := FBu[i,:]
                    ribbon := [-FBl[i,:] + m, FBu[i,:] - m]
                end
                vg[i,:],m
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
                    pi = P[i,:]
                    if bounds
                        ribbon := [-PBl[i,:] + pi, PBu[i,:] - pi]
                    end
                    vg[i,:],pi
                end
            end
        end

    end
    delete!(plotattributes, :phase)
    delete!(plotattributes, :bounds)
    delete!(plotattributes, :nMC)
    delete!(plotattributes, :mcmean)

    nothing

end


@recipe function plot_spectralext(::Type{Val{:spectralext}}, x, y, z)
    xi,w = y.x, y.w
    title --> "Spectrum"
    Nf = length(w)
    x = reshape_params(xi,Nf)
    ax  = abs2.(x)
    px  = angle.(x)
    ax
end
