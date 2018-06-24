using ProximalOperators

"""
`ls_sparse_spectral_lpv(Y,X,V,w,Nv::Int; λ = 1, coulomb = false, normalize=true,
            coulomb    = false,
            normalize  = true,
            iters      = 10000,   # ADMM maximum number of iterations
            tol        = 1e-5,    # ADMM tolerance
            printerval = 100,     # Print this often
            cb(x)      = nothing, # Callback function
            μ          = 0.05`)

Perform LPV spectral estimation using the method presented in
Bagge Carlson et al. "Linear Parameter-Varying Spectral Decomposition."
modified to include a sparsity-promoting L1 group-lasso penalty on the coefficients.
The groups are based on frequency, meaning a solution in which either all parameters
For a particular frequency are zero, or all are non-zero.
This is useful in the identification of frequency components among a large set of possible frequencies.
See the paper or README For additional details.

`Y` output\n
`X` sample locations\n
`V` scheduling signal\n
`w` frequency vector\n
`Nv` number of basis functions\n
`λ` Regularization parameter\n
`coulomb` Assume discontinuity at `v=0` (useful For signals where, e.g., Coulomb friction might cause issues.)\n
`normalize` Use normalized basis functions (See paper For details).


See also `psd`, `ls_spectral_lpv` and `ls_windowpsd_lpv`
"""
function ls_sparse_spectral_lpv(y::AbstractVector, X::AbstractVector, V::AbstractVector,
    w, Nv::Integer;
    coulomb = false,
    normalize=true,
    iters      = 10000,
    tol        = 1e-5,
    printerval = 100,
    cb         = nothing,
    λ          = 1,
    μ          = 0.05,
    kwargs...)


    w        = w[:]
    T        = length(Y)
    Nf       = length(w)
    K        = LPVSpectral.basis_activation_func(V,Nv,normalize,coulomb)
    M(w,X,V) = vec(vec(exp.(im*w.*X))*K(V)')'
    As       = zeros(Complex128,N,Nf*Nv)

    for n = 1:T
        As[n,:] = M(w,X[n],V[n])
    end

    params = LPVSpectral.real_complex_bs(As,Y,λ) # Initialize with standard least squares
    inds = reshape(1:2Nf*Nv, Nf, :)'[:] # Permute parameters so that groups are adjacent
    inds = vcat(inds...)
    x      = [real.(params); imag.(params)][inds]
    Φ      = [real.(As) imag.(As)][:,inds]
    e      = Φ*x-Y
    Σ      = var(e)*inv(Φ'Φ + λ*I)

    nparams = size(Φ,2) # 2Nf*Nv
    z       = zeros(size(x))

    @assert 0 ≤ μ ≤ 1 "μ should be ≤ 1"

    Q     = Φ'Φ
    q     = -Φ'y
    proxf = ProximalOperators.QuadraticIterative(2Q,2q)

    gs    = ntuple(f->NormL2(λ), Nf)
    indsg = ntuple(f->((f-1)*2Nv+1:f*2Nv, ) ,Nf)
    proxg = SlicedSeparableSum(gs, indsg)

    u     = zeros(size(z))
    zu    = similar(u)
    xu    = similar(u)
    xz    = similar(u)
    for i = 1:iters

        zu .= z.-u
        prox!(x, proxf, zu, μ)
        xu .= x .+ u
        prox!(z, proxg, xu, μ)
        xz .= x .- z
        u  .+= xz

        nxz = norm(xz)
        if i % printerval == 0
            @printf("%d ||x-z||₂ %.10f\n", i,  nxz)
            if cb != nothing
                cb(x)
            end
        end
        if nxz < tol
            info("||x-z||₂ ≤ tol")
            break
        end
    end
    x = x[sortperm(inds)] # Sortperm is inverse of inds
    params = complex(x[1:end÷2], x[end÷2+1:end])
    LPVSpectral.SpectralExt(y, X, V, w, Nv, λ, coulomb, normalize, params, nothing)
end


# using LPVSpectral, Plots, LaTeXStrings, DSP
#
# function generate_signal(f,w,N, modphase=false)
#     x = sort(10rand(N)) # Sample points
#     v = linspace(0,1,N) # Scheduling variable
#
#     # generate output signal
#     dependence_matrix = Float64[f[(i-1)%length(f)+1](v) for v in v, i in eachindex(w)] # N x nw
#     frequency_matrix  = [cos(w*x -0.5modphase*(dependence_matrix[i,j])) for (i,x) in enumerate(x), (j,w) in enumerate(w)] # N x nw
#     y = sum(dependence_matrix.*frequency_matrix,2)[:] # Sum over all frequencies
#     y += 0.1randn(size(y))
#     y,v,x,frequency_matrix, dependence_matrix
# end
#
# N      = 500 # Number of training data points
# f      = [v->2v^2, v->2/(5v+1), v->3exp(-10*(v-0.5)^2),] # Functional dependences on the scheduling variable
# w      = 2π*[2,10,20] # Frequency vector
# w_test = 2π*(2:2:25) # Test Frequency vector, set w_test = w for a nice Function visualization
#
# Y,V,X,frequency_matrix, dependence_matrix = generate_signal(f,w,N, true)
#
# λ      = 2 # Regularization parameter
# normal = true # Use normalized basis functions
# Nv     = 50   # Number of basis functions
#
# se = ls_sparse_spectral_lpv(Y,X,V,w_test,Nv; λ = λ, normalize = normal, tol=1e-8, printerval=10, iters=6000) # Perform LPV spectral estimation
# se = ls_spectral_lpv(Y,X,V,w_test,Nv; λ = 0.02, normalize = normal)
#
# # plot(X,[Y V], linewidth=[1 2], lab=["\$y_t\$" "\$v_t\$"], xlabel=L"$x$ (sampling points)", title=L"Test signal $y_t$ and scheduling signal $v_t$", legend=true, xlims=(0,10), grid=false, c=[:cyan :blue])
# plot(se; normalization=:none, dims=2, l=:solid, c = [:red :green :blue], fillalpha=0.5, nMC = 5000, fillcolor=[RGBA(1,.5,.5,.5) RGBA(.5,1,.5,.5) RGBA(.5,.5,1,.5)], linewidth=2, bounds=true, lab=reshape(["Est. \$\\omega = $(round(w/π))\\pi \$" for w in w_test],1,:), phase = false)
# plot!(V,dependence_matrix, title=L"Functional dependencies $A(\omega,v)$", xlabel=L"$v$", ylabel=L"$A(\omega,v)$", c = [:red :green :blue], l=:dot, linewidth=2,lab=reshape(["True \$\\omega = $(round(w/π))\\pi\$" for w in w],1,:), grid=false)
#
# # Plot regular spectrum
# spectrum_lpv   = psd(se) # Calculate power spectral density
# fs             = N/(X[end]-X[1]) # This is the (approximate) sampling freqency of the generated signal
# spectrum_per   = DSP.periodogram(Y, fs=fs)
# spectrum_welch = DSP.welch_pgram(Y, fs=fs)
# plot(2π*collect(spectrum_per.freq), spectrum_per.power, lab="Periodogram", l=:path, m=:none, yscale=:log10, c=:cyan)
# plot!(2π*collect(spectrum_welch.freq), spectrum_welch.power, lab="Welch", l=:path, m=:none, yscale=:log10, linewidth=2, c=:blue)
# plot!(w_test,spectrum_lpv/fs, xlabel=L"$\omega$ [rad/s]", ylabel="Spectral density", ylims=(-Inf,Inf), grid=false, lab="LPV", l=:scatter, m=:o, yscale=:log10, c=:orange)
