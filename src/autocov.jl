function autofun(f,t::AbstractVector,h::AbstractVector{<:Vector{T}},maxlag;kwargs...) where T<:Number
    res = map(zip(t,h)) do (t,h)
        f(t,h,maxlag;kwargs...)
    end
    Tau = getindex.(res,1)
    Y = getindex.(res,2)
    τ = reduce(vcat, Tau)
    y = reduce(vcat, Y)

    perm = @views sortperm(τ)
    return τ[perm],y[perm]
end


StatsBase.autocov(t::AbstractVector,h::AbstractVector{<:Vector{T}},maxlag;kwargs...) where T<:Number = autofun(autocov,t,h,maxlag;kwargs...)
StatsBase.autocor(t::AbstractVector,h::AbstractVector{<:Vector{T}},maxlag;kwargs...) where T<:Number = autofun(autocor,t,h,maxlag;kwargs...)

"""
    τ, acf = autocov(t::AbstractVector, y::AbstractVector{T}, maxlag; normalize=false) where T <: Number

Calculate the autocovariance function for a signal `y` sampled at sample times `t`. This is useful if the signal is sampled non-equidistantly.

If `y` and `t` are vectors of vectors, the acf for all data is estimated and concatenated in the output, which is always two vectors of numbers.

The returned vectors are samples of the acf at lags `τ`. `τ` contains differences between entries in `t`, whereas `lags` supplied to the standard metohd of `autocov` are the differences between *indices* in `y` to compare. The result will in general be noisier that the stadard `autocov` and may require further estimation, e.g., by means of gaussian process regression or quantile regression. The following should hold `mean(acf[τ.==0]) ≈ var(y)`.

#Arguments:
- `t`: Vector of sample locations/times
- `y`: signal
- `maxlag`: Specifies the maximum difference between two values in `t` for which to calculate the ACF
- `normalize`: If false (default) the behaviour mimics the standard method from StatsBase and the ACF decreases automatically for higher lags, even if the theoretical ACF does not. If `true`, the theoretical ACF is estimated. `normalize = true` leads to sharper spectral estimates when taking the fft of the ACF. Note that the variance in the estimated ACF will be high for large lags if `true`.
"""
function StatsBase.autocov(t::AbstractVector{tT},y::AbstractVector{T},maxlag::Number; normalize=false) where {T<:Number, tT<:Number}
    isequidistant(t) || return _autocov(t,y,maxlag;normalize=normalize)
    ly = length(y)
    length(t) == ly || throw(ArgumentError("t and y must be the same length"))

    acf = zeros(T, (ly^2+ly)÷2)
    τ = zeros(tT, (ly^2+ly)÷2)
    k = 1
    @inbounds for i in eachindex(y)
        for j in 0:length(y)-1
            i+j > ly && continue
            τi = abs(t[i+j]-t[i])
            τi > maxlag && continue
            τ[k] = τi
            c = dot(y, 1:ly-j, y, 1+j:ly) / (ly-normalize*(j-1)) # the -1 is important, it corresponds to `corrected=true` for `var`. The variance increases for large lags and this becomes extra important close to lag=length(y)-1
            acf[k] = c
            k += 1
        end
    end

    perm = @views sortperm(τ[1:k-1])
    if all(x->x==y[1], y) || var(y) < eps()
        return τ[perm], zeros(T,length(perm))
    end
    return τ[perm],acf[perm]
end


"""
    τ, acf = autocor(t::AbstractVector, y::AbstractVector{T}, maxlag; normalize=false) where T <: Number

Calculate the auto correlation function for a signal `y` sampled at sample times `t`. This is useful if the signal is sampled non-equidistantly.

If `y` and `t` are vectors of vectors, the acf for all data is estimated and concatenated in the output, which is always two vectors of numbers.

The returned vectors are samples of the acf at lags `τ`. `τ` contains differences between entries in `t`, whereas `lags` supplied to the standard metohd of `autocov` are the differences between *indices* in `y` to compare. The result will in general be noisier that the stadard `autocov` and may require further estimation, e.g., by means of gaussian process regression or quantile regression. The following should hold `mean(acf[τ.==0]) ≈ var(y)`.

#Arguments:
- `t`: Vector of sample locations/times
- `y`: signal
- `maxlag`: Specifies the maximum difference between two values in `t` for which to calculate the ACF
- `normalize`: If false (default) the behaviour mimics the standard method from StatsBase and the ACF decreases automatically for higher lags, even if the theoretical ACF does not. If `true`, the theoretical ACF is estimated. `normalize = true` leads to sharper spectral estimates when taking the fft of the ACF. Note that the variance in the estimated ACF will be high for large lags if `true`.
"""
function StatsBase.autocor(t::AbstractVector{tT},y::AbstractVector{T},maxlag::Number; normalize=false) where {T<:Number, tT<:Number}
    isequidistant(t) || return _autocor(t,y,maxlag;normalize=normalize)
    ly = length(y)
    length(t) == ly || throw(ArgumentError("t and y must be the same length"))

    acf = zeros(T, (ly^2+ly)÷2)
    τ = zeros(tT, (ly^2+ly)÷2)
    k = 1
    dd = dot(y,y)
    @inbounds for i in eachindex(y)
        for j in 0:length(y)-1
            i+j > ly && continue
            τi = abs(t[i+j]-t[i])
            τi > maxlag && continue
            τ[k] = τi
            c = dot(y, 1:ly-j, y, 1+j:ly) / (dd*(ly - normalize*(j-0))/ly)
            acf[k] = c
            k += 1
        end
    end
    perm = @views sortperm(τ[1:k-1])
    if dd < eps()
        return τ[perm], ones(T,length(perm))
    end
    return τ[perm],acf[perm]
end


isequidistant(v::Union{<:UnitRange, <:StepRange, <:StepRangeLen}) = step(v) > 0
function isequidistant(v)
    d = v[2]-v[1]
    d > 0 || return false
    d = abs(d)
    for i in 3:length(v)
        abs(abs(v[i]-v[i-1])-d) < 20d*eps() || return false
    end
    true
end





function _autocov(t::AbstractVector{tT},y::AbstractVector{T},maxlag::Number; normalize=false) where {T<:Number, tT<:Number}
    ly = length(y)
    length(t) == ly || throw(ArgumentError("t and y must be the same length"))

    acf = zeros(T, (ly^2+ly)÷2)
    τ = zeros(tT, (ly^2+ly)÷2)
    k = 1
    @inbounds for i in eachindex(y)
        for j in 0:length(y)-1
            i+j > ly && continue
            τi = abs(t[i+j]-t[i])
            τi > maxlag && continue
            τ[k] = τi
            c = y[i]*y[i+j]
            acf[k] = c
            k += 1
        end
    end

    perm = @views sortperm(τ[1:k-1])
    if all(x->x==y[1], y) || var(y) < eps()
        return τ[perm], zeros(T,length(perm))
    end
    return τ[perm],acf[perm]
end

function _autocor(t::AbstractVector{tT},y::AbstractVector{T},maxlag::Number; normalize=false) where {T<:Number, tT<:Number}
    ly = length(y)
    length(t) == ly || throw(ArgumentError("t and y must be the same length"))

    acf = zeros(T, (ly^2+ly)÷2)
    τ = zeros(tT, (ly^2+ly)÷2)
    k = 1
    dd = var(y)
    @inbounds for i in eachindex(y)
        for j in 0:length(y)-1
            i+j > ly && continue
            τi = abs(t[i+j]-t[i])
            τi > maxlag && continue
            τ[k] = τi
            c = y[i]*y[i+j]/dd
            acf[k] = c
            k += 1
        end
    end
    perm = @views sortperm(τ[1:k-1])
    if dd < eps()
        return τ[perm], ones(T,length(perm))
    end
    zeroinds = τ .== 0
    acf[zeroinds] .= 1
    return τ[perm],acf[perm]
end
