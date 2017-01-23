module LPVSpectral
using DSP
using Plots

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
ls_spectralext


# Helper functions
# export SE, fourier, manhattan, plot

export ComplexNormal, SpectralExt


# ComplexNormal
export cn_V2ΓC,cn_V2ΓC,cn_Vxx,cn_Vyy,cn_Vxy,cn_Vyx,cn_fVxx,cn_fVyy,cn_fVxy,
cn_fVyx,cn_Vs,cn_V,cn_fV,Σ,pdf,affine_transform, rand


end # module
