using LPVSpectral, Plots, LaTeXStrings, DSP

function generate_signal(f,w,N, modphase=false)
    x = sort(10rand(N)) # Sample points
    v = linspace(0,1,N) # Scheduling variable

    # generate output signal
    # phase_matrix
    dependence_matrix = Float64[f[(i-1)%length(f)+1](v) for v in v, i in eachindex(w)] # N x nw
    frequency_matrix = [cos(w*x -0.5modphase*(dependence_matrix[i,j])) for (i,x) in enumerate(x), (j,w) in enumerate(w)] # N x nw
    y = sum(dependence_matrix.*frequency_matrix,2)[:] # Sum over all frequencies
    y += 0.1randn(size(y))
    y,v,x,frequency_matrix, dependence_matrix
end

N = 1000 # Number of training data points
f = [v->2v^2, v->2/(5v+1), v->3exp(-10*(v-0.5)^2),] # Functional dependences on the scheduling variable
w = 2π*[2,10,20] # Frequency vector
w_test = 2π*(2:2:25) # Test Frequency vector, set w_test = w for a nice function visualization

Y,V,X,frequency_matrix, dependence_matrix = generate_signal(f,w,N, true)

# Options for spectral estimation
λ = 0.02        # Regularization parmater
normal = true   # Use normalized basis functions
Nv = 10         # Number of basis functions

se = ls_spectralext(Y,X,V,w_test,Nv; λ = λ, normalize = normal)
psd = ls_windowpsd_ext(Y,X,V,w_test,Nv; λ = λ, normalize = normal)

plot(X,[Y V], linewidth=[1 2], lab=["\$y_t\$" "\$v_t\$"], xlabel=L"$x$ (sampling points)", title=L"Test signal $y_t$ and scheduling signal $v_t$", legend=true, xlims=(0,10), grid=false, c=[:cyan :blue])
plot(se; normalization=:none, dims=2, l=:solid, c = [:red :green :blue], fillalpha=0.5, nMC = 5000, fillcolor=[RGBA(1,.5,.5,.5) RGBA(.5,1,.5,.5) RGBA(.5,.5,1,.5)], linewidth=2, bounds=true, lab=["Est. \$\\omega = $(round(w/π))\\pi \$" for w in w]', phase = false)
plot!(V,dependence_matrix, title=L"Functional dependeces $A(\omega,v)$", xlabel=L"$v$", ylabel=L"$A(\omega,v)$", c = [:red :green :blue], l=:dot, linewidth=2,lab=["True \$\\omega = $(round(w/π))\\pi\$" for w in w]', grid=false)


Nf = length(w_test)
rp = LPVSpectral.reshape_params(copy(se.x),Nf)
spectrum_ext  = sum(rp,2) |> abs2 # See paper for details
fs = N/(X[end]-X[1]) # This is the (approximate) sampling freqency of the generated signal
spectrum_per = DSP.periodogram(Y, fs=fs)
spectrum_welch = DSP.welch_pgram(Y, fs=fs)
plot(2π*collect(spectrum_per.freq), spectrum_per.power, lab="Periodogram", l=:path, m=:none, yscale=:log10, c=:cyan)
plot!(2π*collect(spectrum_welch.freq), spectrum_welch.power, lab="Welch", l=:path, m=:none, yscale=:log10, linewidth=2, c=:blue)
plot!(w_test,spectrum_ext/fs, xlabel=L"$\omega$ [rad/s]", ylabel="Spectral density", ylims=(-Inf,Inf), grid=false, lab="LPV", l=:scatter, m=:o, yscale=:log10, c=:orange)
plot!(w_test,psd/fs, xlabel=L"$\omega$ [rad/s]", ylabel="Spectral density", ylims=(-Inf,Inf), grid=false, lab="Windowed LPV", l=:scatter, m=:o, yscale=:log10, c=:magenta)
