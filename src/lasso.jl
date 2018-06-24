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
    T        = length(y)
    Nf       = length(w)
    K        = LPVSpectral.basis_activation_func(V,Nv,normalize,coulomb)
    M(w,X,V) = vec(vec(exp.(im*w.*X))*K(V)')'
    As       = zeros(Complex128,T,Nf*Nv)

    for n = 1:T
        As[n,:] = M(w,X[n],V[n])
    end

    params = LPVSpectral.real_complex_bs(As,y,λ) # Initialize with standard least squares
    inds   = reshape(1:2Nf*Nv, Nf, :)'[:]        # Permute parameters so that groups are adjacent
    inds   = vcat(inds...)
    x      = [real.(params); imag.(params)][inds]
    Φ      = [real.(As) imag.(As)][:,inds]
    e      = Φ*x-y
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
