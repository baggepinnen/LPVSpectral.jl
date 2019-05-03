function detrend!(x::Vector, order=0, t = 1:length(x))
    x[:] .-= mean(x)
    if order == 1
        k = x\t
        x[:] .-= k*t
    end
    x
end

"""detrend(x, order=0, t = 1:length(x))
Removes the trend of order `order`, i.e, mean and, if order=1, the slope of the signal `x`
If `order` = 1, then the sampling points of `x` can be supplied as `t` (default: `t = 1:length(x)`)
"""
function detrend(x,args...)
    y = copy(x)
    detrend!(y,args...)
end

"""basis_activation_func(V,Nv,normalize,coulomb)

Returns a func v->ϕ(v) ∈ ℜ(Nv) that calculates the activation of `Nv` basis functions spread out to cover V nicely. If coulomb is true, then we get twice the number of basis functions, 2Nv
"""
function basis_activation_func(V,Nv,normalize,coulomb)
    if coulomb # If Coulomb setting is activated, double the number of basis functions and clip the activation at zero velocity (useful for data that exhibits a discontinuity at v=0, like coulomb friction)
        vc      = range(0, stop=maximum(abs.(V)), length=Nv+2)
        vc      = vc[2:end-1]
        vc      = [-vc[end:-1:1]; vc]
        Nv      = 2Nv
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K       = normalize ? V -> _Kcoulomb_norm(V,vc,gamma) : V -> _Kcoulomb(V,vc,gamma) # Use coulomb basis function instead
    else
        vc      = range(minimum(V), stop=maximum(V), length=Nv)
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K       = normalize ? V -> _K_norm(V,vc,gamma) : V -> _K(V,vc,gamma)
    end
end


"""`ridgereg(A,b,λ)`
Accepts `λ` to solve the ridge regression problem using the formulation `[A;λI]\\[b;0]. λ should be given with the same dimension as the columns of A, i.e. if λ represents an inverse standard deviation, then 1/λ = σ, not 1/λ = σ²`"""
function ridgereg(A,b,λ)
    n = size(A,2)
    [A; λ*I]\[b;zeros(n)]
end

"""`real_complex_bs(A,b, λ=0)`
Replaces the backslash operator For complex arguments. Expands the A-matrix into `[real(A) imag(A)]` and performs the computation using real arithmetics. Optionally accepts `λ` to solve the ridge regression problem using the formulation `[A;λI]\\[b;0]. λ should be given with the same dimension as the columns of A, i.e. if λ represents a standard deviation, then λ = σ, not λ = σ²`
"""
function real_complex_bs(A,b, λ=0)
    n  = size(A,2)
    Ar = [real(A) imag(A)]
    xr = λ > 0 ? [Ar; λ*I]\[b;zeros(2n)] : Ar\b
    x  = complex.(xr[1:n], xr[n+1:end])
end

function fourier_solve(A,y,zerofreq,λ=0)
    n  = size(A,2)
    x = λ > 0 ? [A; λ*I]\[y;zeros(n)] : A\y
    fourier2complex(x,zerofreq)
end

function fourier2complex(x,zerofreq)
    n = length(x)÷2
    if zerofreq === nothing
        return complex.(x[1:n], x[n+1:end])
    else
        x0 = x[zerofreq]
        x = deleteat!(copy(x), zerofreq)
        x = complex.(x[1:n], x[n+1:end])
        insert!(x,zerofreq,x0)
        return x
    end
end


""" Returns params as a [nω × N] matrix"""
reshape_params(x,Nf) = reshape(x, Nf,round(Int,length(x)/Nf))


## Complex Normal
import Base.rand

mutable struct ComplexNormal{T<:Complex}
    m::AbstractVector{T}
    Γ::Cholesky
    C::Symmetric{T}
end

function ComplexNormal(X::AbstractVecOrMat,Y::AbstractVecOrMat)
    @assert size(X) == size(Y)
    mc  = complex.(mean(X,dims=1)[:], mean(Y,dims=1)[:])
    V   = Symmetric(cov([X Y]))
    Γ,C = cn_V2ΓC(V)
    ComplexNormal(mc,Γ,C)
end

function ComplexNormal(X::AbstractVecOrMat{T}) where T<:Complex
    ComplexNormal(real.(X),imag.(X))
end

function ComplexNormal(m::AbstractVector{T},V::AbstractMatrix{T}) where T<:Real
    n   = Int(length(m)/2)
    mc  = complex.(m[1:n], m[n+1:end])
    Γ,C = cn_V2ΓC(V)
    ComplexNormal(mc,Γ,C)
end

function ComplexNormal(mc::AbstractVector{Tc},V::AbstractMatrix{Tr}) where {Tr<:Real, Tc<:Complex}
    Γ,C = cn_V2ΓC(V)
    ComplexNormal(mc,Γ,C)
end

function cn_V2ΓC(V::Symmetric{T}) where T<:Real
    n   = size(V,1)÷2
    Vxx = V[1:n,1:n]
    Vyy = V[n+1:end,n+1:end]
    Vxy = V[1:n,n+1:end]
    Vyx = V[n+1:end,1:n]
    Γ   = cholesky(complex.(Vxx + Vyy, Vyx - Vxy))
    C   = Symmetric(complex.(Vxx - Vyy, Vyx + Vxy))
    Γ,C
end

cn_V2ΓC(V::AbstractMatrix{T}) where {T<:Real} = cn_V2ΓC(Symmetric(V))

@inline cn_Vxx(Γ,C) = cholesky(real.(Matrix(Γ)+C)/2)
@inline cn_Vyy(Γ,C) = cholesky(real.(Matrix(Γ)-C)/2)
@inline cn_Vxy(Γ,C) = cholesky(imag.(-Matrix(Γ)+C)/2)
@inline cn_Vyx(Γ,C) = cholesky(imag.(Matrix(Γ)+C)/2)

@inline cn_fVxx(Γ,C) = real.(Matrix(Γ)+C)/2
@inline cn_fVyy(Γ,C) = real.(Matrix(Γ)-C)/2
@inline cn_fVxy(Γ,C) = imag.(-Matrix(Γ)+C)/2
@inline cn_fVyx(Γ,C) = imag.(Matrix(Γ)+C)/2

@inline cn_Vs(Γ,C) = cn_Vxx(Γ,C),cn_Vyy(Γ,C),cn_Vxy(Γ,C),cn_Vyx(Γ,C)
@inline cn_fV(Γ,C) = [cn_fVxx(Γ,C) cn_fVxy(Γ,C); cn_fVyx(Γ,C) cn_fVyy(Γ,C)]
@inline cn_V(Γ,C) = cholesky(cn_fV(Γ,C))
@inline Σ(cn::ComplexNormal) = Matrix(cn.Γ) # TODO: check this

for f in [:cn_Vxx,:cn_Vyy,:cn_Vxy,:cn_Vyx,:cn_fVxx,:cn_fVyy,:cn_fVxy,:cn_fVyx,:cn_Vs,:cn_V,:cn_fV]
    @eval ($f)(cn::ComplexNormal) = ($f)(cn.Γ,cn.C)
end

"""
`f(cn::ComplexNormal, z)`

Probability density Function `f(z)` for a complex normal distribution.
This can probably be more efficiently implemented
"""
function pdf(cn::ComplexNormal, z)
    k = length(cn.m)
    R = conj(cn.C)'*inv(cn.Γ)
    P = Matrix(cn.Γ)-R*cn.C # conj(Γ) = Γ for Γ::Cholesky
    cm = conj(cn.m)
    cz = conj(z)
    zmm = z-cn.m
    czmm = cz-cm
    ld = [czmm' zmm']
    rd = [zmm; czmm]
    S = [Matrix(cn.Γ) cn.C;conj(cn.C) Matrix(cn.Γ)] # conj(Γ) = Γ for Γ::Cholesky
    1/(π^k*sqrt(det(cn.Γ)*det(P))) * exp(-0.5* ld*(S\rd))
end

affine_transform(cn::ComplexNormal, A,b) = ComplexNormal(A*cn.m+b, cholesky(A*Matrix(cn.Γ)*conj(A')), Symmetric(A*cn.C*A'))


function rand(cn::ComplexNormal,s::Integer)
    L = cn_V(cn).U
    m = [real(cn.m); imag(cn.m)]
    n = length(cn.m)
    z = (m' .+ randn(s,2n)*L)
    return complex.(z[:,1:n],z[:,n+1:end])
end
