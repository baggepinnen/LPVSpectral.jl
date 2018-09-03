"""
Least-squares spectral estimation toolbox.

For help, see README.md at https://github.com/baggepinnen/LPVSpectral.jl and

[Fredrik Bagge Carlson, Anders Robertsson, Rolf Johansson: "Linear Parameter-Varying Spectral Decomposition". In: 2017 American Control Conference 2017.]
(http://lup.lub.lu.se/record/ac32368e-e199-44ff-b76a-36668ac7d595)
Available at: http://lup.lub.lu.se/record/ac32368e-e199-44ff-b76a-36668ac7d595

This module provide the functions
```
ls_spectral
tls_spectral
ls_windowpsd
ls_windowcsd
ls_cohere
ls_spectral_lpv
ls_sparse_spectral_lpv
ls_windowpsd_lpv
basis_activation_func
```
and re-exports the following from DSP.jl
```
export periodogram, welch_pgram, Windows
```
Periodogram types and SpectralExt type can be plotted using `plot(x::SpectralExt)`
"""
module LPVSpectral
using LinearAlgebra, Statistics, Printf
using DSP
using Plots
using ProximalOperators

"""
LPV Spectral estimation result type.

See `ls_spectral_lpv` for additional help.

An object of this type can be plotted if `Plots.jl` is installed. Use regular Plots-syntax, with the additional attributes
```
normalization= :none / :sum / :max
normdim = :freq / :v # Only applies if normalization= :sum or :max
dims = 2 or 3 (default = 2)
```

Fields:
```
Y::AbstractVector
X::AbstractVector
V::AbstractVector
w
Nv
λ
coulomb::Bool
normalize::Bool
x                   # The estimated parameters
Σ                   # Covariance of the estimated parameters
```
"""
struct SpectralExt
    Y::AbstractVector
    X::AbstractVector
    V::AbstractVector
    w
    Nv
    λ
    coulomb::Bool
    normalize::Bool
    x
    Σ
end


include("utilities.jl")
include("windows.jl")
include("plotting.jl")
include("lsfft.jl")
include("lasso.jl")

# Functions
export ls_spectral,
tls_spectral,
ls_sparse_spectral,
ls_windowpsd,
ls_windowcsd,
ls_cohere,
ls_spectral_lpv,
ls_sparse_spectral_lpv,
ls_windowpsd_lpv,
basis_activation_func,
SpectralExt,
psd,
detrend,
detrend!

# Re-export
export plot,
periodogram,
welch_pgram,
Windows


# ComplexNormal
export ComplexNormal

export cn_V2ΓC,cn_V2ΓC,cn_Vxx,cn_Vyy,cn_Vxy,cn_Vyx,cn_fVxx,cn_fVyy,cn_fVxy,
cn_fVyx,cn_Vs,cn_V,cn_fV,Σ,pdf,affine_transform, rand


end # module
