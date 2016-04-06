module LPVSpectral
using DSP
using Plots


include("utilities.jl")
include("lsfft.jl")
include("gp_spectral.jl")


# Functions
export ls_spectral,
ls_spectral_real,
tls_spectral_real,
ls_spectral,
ls_spectral_real,
lswindowpsd,
lswindowcsd,
lscohere,
ls_spectralext,
GP_spectral

# Types
export GPcov,
GPfreq,
GPspectralOpts,
GPspectum

# Helper functions
export SE, fourier, manhattan, plot




end # module
