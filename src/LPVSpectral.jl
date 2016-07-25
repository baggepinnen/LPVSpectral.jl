module LPVSpectral
using DSP
using Plots


include("utilities.jl")
include("lsfft.jl")
include("gp_spectral.jl")


# Functions
export ls_spectral,
lswindowpsd,
lswindowcsd,
lscohere,
ls_spectralext,
GP_spectral

# Types
export GPcov,
GPfreq,
GPspectralOpts,
GPspectum,
ComplexNormal

# Helper functions
export SE, fourier, manhattan, plot

# ComplexNormal
export cn_V2ΓC,cn_V2ΓC,cn_Vxx,cn_Vyy,cn_Vxy,cn_Vyx,cn_fVxx,cn_fVyy,cn_fVxy,
cn_fVyx,cn_Vs,cn_V,cn_fV,Σ,pdf,affine_transform, rand


end # module
