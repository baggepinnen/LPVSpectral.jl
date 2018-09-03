using LPVSpectral, Plots, LaTeXStrings, DSP, ProximalOperators
gr()

function generate_signal(f,w,N, modphase=false)
    x = sort(10rand(N)) # Sample points
    v = range(0, stop=1, length=N) # Scheduling variable

    # generate output signal
    dependence_matrix = Float64[f[(i-1)%length(f)+1](v) for v in v, i in eachindex(w)] # N x nw
    frequency_matrix  = [cos(w*x -0.5modphase*(dependence_matrix[i,j])) for (i,x) in enumerate(x), (j,w) in enumerate(w)] # N x nw
    y = sum(dependence_matrix.*frequency_matrix,dims=2)[:] # Sum over all frequencies
    y += 0.1randn(size(y))
    y,v,x,frequency_matrix, dependence_matrix
end

N      = 500 # Number of training data points
f      = [v->2v^2, v->2/(5v+1), v->3exp(-10*(v-0.5)^2),] # Functional dependences on the scheduling variable
w      = 2π*[2,10,20] # Frequency vector
w_test = 2π*(2:2:25) # Test Frequency vector, set w_test = w for a nice Function visualization

Y,V,X,frequency_matrix, dependence_matrix = generate_signal(f,w,N, true)

λ      = 0.02 # Regularization parameter
λs     = 5    # Regularization parameter group-lasso
normal = true # Use normalized basis functions
Nv     = 50   # Number of basis functions

zeronan(x) = ifelse(x==0, NaN, x)
callback(x,z) = ()#plot(max.(abs2.([complex.(x[1:end÷2], x[end÷2+1:end]) complex.(z[1:end÷2], z[end÷2+1:end])]), 1e-20), reuse=true, show=true)

ses = ls_sparse_spectral_lpv(Y,X,V,w_test,Nv; λ = λs, normalize = normal, tol=1e-8, printerval=10, iters=2000, cb=callback) # Perform LPV spectral estimation
se  = ls_spectral_lpv(Y,X,V,w_test,Nv; λ = 0.02, normalize = normal)
xs  = LPVSpectral.ls_sparse_spectral(Y,X,1:0.1:25; λ=20, tol=1e-9, printerval=2, iters=9000, μ=0.00001,cb=callback)
# xsi = LPVSpectral.ls_sparse_spectral(Y,X,1:0.1:25; proxg=IndBallL0(6), λ=0.5, tol=1e-9, printerval=100, iters=9000, μ=0.0001,cb=callback)
xsw = ls_windowpsd(Y,X,1:0.5:22; nw=2, estimator=ls_sparse_spectral, λ=0.2, tol=1e-10, printerval=10000, iters=60000, μ=0.0001)

# plot(X,[Y V], linewidth=[1 2], lab=["\$y_t\$" "\$v_t\$"], xlabel=L"$x$ (sampling points)", title=L"Test signal $y_t$ and scheduling signal $v_t$", legend=true, xlims=(0,10), grid=false, c=[:cyan :blue])
plot(se; normalization=:none, dims=2, l=:solid, c = :orange, fillalpha=0.5, nMC = 100, fillcolor=:orange, linewidth=2, bounds=true, lab=reshape(["Est. \$\\omega = $(round(w/π))\\pi \$" for w in w_test],1,:), phase = false)
plot!(ses; normalization=:none, dims=2, l=:solid, c = :green, linewidth=2, lab=reshape(["Est. \$\\omega = $(round(w/π))\\pi \$" for w in w_test],1,:), phase = false)
plot!(V,dependence_matrix, title=L"Functional dependencies $A(\omega,v)$", xlabel=L"$v$", ylabel=L"$A(\omega,v)$", c = [:blue], l=:dot, linewidth=2,lab=reshape(["True \$\\omega = $(round(w/π))\\pi\$" for w in w],1,:), grid=false)

## Plot regular spectrum
spectrum_lpv   = psd(se) # Calculate power spectral density
spectrum_lpvs  = psd(ses) # Calculate sparse power spectral density
fs             = N/(X[end]-X[1]) # This is the (approximate) sampling freqency of the generated signal
spectrum_per   = DSP.periodogram(Y, fs=fs)
spectrum_welch = DSP.welch_pgram(Y, fs=fs)
plot(2π*collect(spectrum_per.freq), spectrum_per.power, lab="Periodogram", l=:path, m=:none, yscale=:log10, c=:cyan, legend=:bottomright)
plot!(2π*collect(spectrum_welch.freq), spectrum_welch.power, lab="Welch", l=:path, m=:none, yscale=:log10, linewidth=2, c=:blue)
plot!(2π*(1:0.1:25), zeronan.(abs2.(xs)), lab="sparse", l=:path, m=:none, yscale=:log10, linewidth=2, c=:magenta)
plot!(w_test,spectrum_lpv/fs, xlabel=L"\omega [rad/s]", ylabel="Spectral density", ylims=(-Inf,Inf), grid=false, lab="LPV", l=:scatter, m=:o, yscale=:log10, c=:orange)
plot!(w_test,zeronan.(spectrum_lpvs)./fs, lab="Sparse LPV", l=:scatter, m=:x, c=:green)
# plot!(2π*(1:0.1:25), max.(abs2.(xsi), 1e-15), lab="sparse ind ball", l=:path, m=:none, yscale=:log10, linewidth=2, c=:yellow)
# plot!(2π*(1:0.5:22), zeronan.(abs2.(xsw)), lab="sparse windowed", l=:path, m=:none, yscale=:log10, linewidth=2, c=:orange)
# savetikz("/local/home/fredrikb/phdthesis/spectralest/figs/spectrum_gen2.tex")
