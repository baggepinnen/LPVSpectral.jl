using ProximalOperators

"""
`ls_sparse_spectral_lpv(Y,X,V,w,Nv::Int; λ = 1, coulomb = false, normalize=true,
coulomb    = false,
normalize  = true)

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

See `?ADMM` for keyword arguments to control the solver.

See also `psd`, `ls_spectral_lpv` and `ls_windowpsd_lpv`
"""
function ls_sparse_spectral_lpv(y::AbstractVector, X::AbstractVector, V::AbstractVector,
    w, Nv::Integer;
    λ         = 1,
    coulomb   = false,
    normalize = true,
    kwargs...)


    w        = w[:]
    T        = length(y)
    Nf       = length(w)
    K        = basis_activation_func(V,Nv,normalize,coulomb)
    M(w,X,V) = vec(vec(exp.(im.*w.*X))*K(V)')'
    As       = zeros(ComplexF64,T,ifelse(coulomb,2,1)*Nf*Nv)

    for n = 1:T
        As[n,:] = M(w,X[n],V[n])
    end

    x      = zeros(2size(As,2))                 # Initialize with standard least squares
    inds   = reshape(1:length(x), Nf, :)'[:] # Permute parameters so that groups are adjacent
    inds   = vcat(inds...)
    # x    = [real.(params); imag.(params)][inds]
    Φ      = [real.(As) imag.(As)][:,inds]
    proxf  = ProximalOperators.LeastSquares(Φ,y,iterative=true)

    gs     = ntuple(f->NormL2(λ), Nf)
    indsg  = ntuple(f->((f-1)*2Nv+1:f*2Nv, ) ,Nf)
    proxg  = SlicedSeparableSum(gs, indsg)

    x,z      = ADMM(x, proxf, proxg; kwargs...)
    z      = z[sortperm(inds)]            # Sortperm is inverse of inds
    params = complex.(z[1:end÷2], z[end÷2+1:end])
    SpectralExt(y, X, V, w, Nv, λ, coulomb, normalize, params, nothing)
end



"""`x,f = ls_sparse_spectral(y,t,f=default_freqs(t), [window::AbstractVector]; λ=1,
proxg      = ProximalOperators.NormL1(λ),
kwargs...)`

perform spectral estimation using the least-squares method with (default) a L1-norm penalty on the
Fourier coefficients, change kwarg `proxg` to e.g. `NormL0(λ)` for a different behavior or ` proxg = IndBallL0(4)` if the number of frequencies is known in advance. Promotes a sparse spectrum. See `?ADMM` for keyword arguments to control the solver.

`y` is the signal to be analyzed
`t` is the sampling points
`f` is a vector of frequencies
"""
function ls_sparse_spectral(y,t,f=default_freqs(t);
    init       = false,
    λ          = 1,
    proxg      = NormL1(λ),
    kwargs...)

    A,zerofreq  = get_fourier_regressor(t,f)
    params = init ? fourier_solve(A,y,λ) : fill(0., length(f)) # Initialize with standard least squares
    x      = [real.(params); imag.(params)]
    proxf  = ProximalOperators.LeastSquares(A,y, iterative=true)
    x,z    = ADMM(x, proxf, proxg; kwargs...)
    params = fourier2complex(z, zerofreq)
    params, f
end


function ls_sparse_spectral(y,t,f, W;
    λ          = 1,
    proxg      = NormL1(λ),
    kwargs...)

    Φ,zerofreq  = get_fourier_regressor(t,f)
    x      = zeros(size(Φ,2))
    Wd     = Diagonal(W)
    Q      = Φ'Wd*Φ
    q      = -Φ'Wd*y
    proxf  = ProximalOperators.QuadraticIterative(2Q,2q)
    x,z = ADMM(x, proxf, proxg; kwargs...)
    params = fourier2complex(z, zerofreq)
    params, f
end

"""
    ADMM(x,proxf,proxg;
    iters      = 10000,   # ADMM maximum number of iterations
    tol        = 1e-5,    # ADMM tolerance
    printerval = 100,     # Print this often
    cb(x,z)    = nothing, # Callback function
    μ          = 0.05`)   # ADMM tuning parameter. If results oscillate, lower this value.
"""
function ADMM(x,proxf,proxg;
    iters      = 10000,
    tol        = 1e-9,
    printerval = 100,
    cb         = nothing,
    μ          = 0.05)

    @assert 0 ≤ μ ≤ 1 "μ should be ≤ 1"

    z     = copy(x)
    u     = zeros(size(x))
    tmp   = similar(u)
    for i = 1:iters
        tmp .= z.-u
        prox!(x, proxf, tmp, μ)
        tmp .= x .+ u
        prox!(z, proxg, tmp, μ)
        tmp .= x .- z
        u  .+= tmp

        nxz = norm(tmp)
        if i % printerval == 0
            @printf("%d ||x-z||₂ %.10f\n", i,  nxz)
            if cb != nothing
                cb(x,z)
            end
        end
        if nxz < tol
            @printf("%d ||x-z||₂ %.10f\n", i,  nxz)
            @info("||x-z||₂ ≤ tol")
            break
        end
    end
    x,z
end
