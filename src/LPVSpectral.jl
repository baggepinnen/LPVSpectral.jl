module LPVSpectral
using DSP
using Plots

"""
LPV Spectral estimation result type.

See `ls_spectralext` for additional help.

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
immutable SpectralExt
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
    fve::Float64
end


include("utilities.jl")
include("plotting.jl")
include("lsfft.jl")


# Functions
export ls_spectral,
tls_spectral,
ls_windowpsd,
ls_windowcsd,
ls_cohere,
ls_spectralext,
ls_windowpsd_ext,
basis_activation_func


# Helper functions
# export SE, fourier, manhattan, plot

export ComplexNormal, SpectralExt


# ComplexNormal
export cn_V2ΓC,cn_V2ΓC,cn_Vxx,cn_Vyy,cn_Vxy,cn_Vyx,cn_fVxx,cn_fVyy,cn_fVxy,
cn_fVyx,cn_Vs,cn_V,cn_fV,Σ,pdf,affine_transform, rand


end # module
