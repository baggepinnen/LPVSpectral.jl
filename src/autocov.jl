function autofun(f,t::AbstractVector,h::AbstractVector{<:Vector{T}},lags) where T<:Number
    res = map(zip(t,h)) do (t,h)
        f(t,h,lags)
    end
    Tau = getindex.(res,1)
    Y = getindex.(res,2)
    τ = reduce(vcat, Tau)
    y = reduce(vcat, Y)
    return τ,y
end


StatsBase.autocov(t::AbstractVector,h::AbstractVector{<:Vector{T}},lags) where T<:Number = autofun(autocov,t,h,lags)
StatsBase.autocor(t::AbstractVector,h::AbstractVector{<:Vector{T}},lags) where T<:Number = autofun(autocor,t,h,lags)

"""
    τ, acf = autocov(t::AbstractVector, y::AbstractVector{T}, lags) where T <: Number

Calculate the autocovariance function for a signal `y` sampled at sample times `t`. This is useful if the signal is sampled non-equidistantly.

If `y` and `t` are vectors of vectors, the acf for all data is estimated and concatenated in the output, which is always two vectors of numbers.

The returned vectors are samples of the acf at lags `τ`. The result will in general be very noisy and require further estimation, e.g., by means of gaussian process regression or quantile regression. The following should hold `mean(acf[τ.==0]) ≈ var(y)`.

#Arguments:
- `t`: Vector of sample locations/times
- `y`: signal
- `lags`: Specifies for which lags to calculate the ACF
"""
function StatsBase.autocov(t::AbstractVector,y::AbstractVector{T},lags::AbstractVector{<:Integer}) where T<:Number

    ly = length(y)
    acf = zeros(T, length(lags)*ly)
    τ = zeros(eltype(t), length(lags)*ly)
    k = 1

    for i in eachindex(y)
        for j in lags
            i+j > ly && continue
            c = dot(y, 1:ly-j, y, 1+j:ly) / ly
            acf[k] = c
            τ[k] = t[i+j]-t[i]
            k += 1
        end
    end
    perm = @views sortperm(τ[1:k-1])
    return τ[perm],acf[perm]
end


"""
    τ, acf = autocor(t::AbstractVector, y::AbstractVector{T}, lags) where T <: Number

Calculate the auto correlation function for a signal `y` sampled at sample times `t`. This is useful if the signal is sampled non-equidistantly.

If `y` and `t` are vectors of vectors, the acf for all data is estimated and concatenated in the output, which is always two vectors of numbers.

The returned vectors are samples of the acf at lags `τ`. The result will in general be very noisy and require further estimation, e.g., by means of gaussian process regression or quantile regression. The following should hold `mean(acf[τ.==0]) ≈ var(y)`.

#Arguments:
- `t`: Vector of sample locations/times
- `y`: signal
- `lags`: Specifies for which lags to calculate the ACF
"""
function StatsBase.autocor(t::AbstractVector,y::AbstractVector{T},lags::AbstractVector{<:Integer}) where T<:Number

    ly = length(y)
    acf = zeros(T, length(lags)*ly)
    τ = zeros(eltype(t), length(lags)*ly)
    k = 1
    dd = dot(y,y)
    for i in eachindex(y)
        for j in lags
            i+j > ly && continue
            c = dot(y, 1:ly-j, y, 1+j:ly) / dd
            acf[k] = c
            τ[k] = t[i+j]-t[i]
            k += 1
        end
    end
    perm = @views sortperm(τ[1:k-1])
    return τ[perm],acf[perm]
end
