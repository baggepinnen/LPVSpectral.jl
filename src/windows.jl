abstract type AbstractWindows end
Base.length(w::AbstractWindows) = length(w.ys);

Base.collect(W::AbstractWindows) = [copy.(w) for w in W]

# Window2 ==========================================================
struct Windows2 <: AbstractWindows
    y::AbstractVector
    t::AbstractVector
    ys
    ts
    W
    """
        Windows2(y, t, n=length(y)÷8, noverlap=-1, window_func=rect)

    noverlap = -1 sets the noverlap to n/2

    Iterating `w::Windows2` produces `(y,t)` of specified length. The window is *not* applied to the signal, instead the window array is available as `w.W`.

    # Arguments:
    - `y`: Signal
    - `t`: Time vector
    - `n`: Number of datapoints per window
    - `noverlap`: Overlap between windows
    - `window_func`: Function to create a vector of weights, e.g., `DSP.hanning, DSP.rect` etc.
    """
    function Windows2(y,t,n::Int=length(y)>>3, noverlap::Int=n>>1,window_func=rect)

        noverlap < 0 && (noverlap = n>>1)
        N   = length(y)
        @assert N == length(t) "y and t has to be the same length"
        W = window_func(n)
        ys = arraysplit(y,n,noverlap)
        ts = arraysplit(t,n,noverlap)
        new(y,t,ys,ts,W)
    end
end

function Base.iterate(w::Windows2, state=1)
    state > length(w) && return nothing
    ((w.ys[state],w.ts[state]), state+1)
end

"""
    mapwindows(f, W::AbstractWindows)

Apply a Function `f` over all windows represented by `W::Windows2`.
`f` must take `(y,t)->ŷ` where `y` and `ŷ` have the same length.
"""
function mapwindows(f::Function, W::AbstractWindows)
    res = map(f,W)
    merge(res,W)
end

mapwindows(f::Function, args...) = mapwindows(f, Windows2(args...))

function Base.merge(yf::AbstractVector{<:AbstractVector},w::Windows2)
    ym = zeros(eltype(yf[1]), length(w.y))
    counts = zeros(Int, length(w.y))
    dpw = length(w.ys[1])
    inds =  1:dpw

    for i in eachindex(w.ys)
        ym[inds] .+= yf[i]
        counts[inds] .+= 1
        inds = inds .+ (dpw-w.ys.noverlap)
        inds =  inds[1]:min(inds[end], length(ym))
    end
    ym ./= max.(counts,1)
end

# Window3 ==========================================================
"""
    Windows3(y, t, v, n=length(y)÷8, noverlap=-1, window_func=rect)

noverlap = -1 sets the noverlap to n/2

#Arguments:
- `y`: Signal
- `t`: Time vector
- `v`: Auxiliary vector
- `n`: Number of datapoints per window
- `noverlap`: Overlap between windows
- `window_func`: Function to apply over window
"""
struct Windows3 <: AbstractWindows
    y::AbstractVector
    t::AbstractVector
    v::AbstractVector
    ys
    ts
    vs
    W
    function Windows3(y::AbstractVector,t::AbstractVector,v::AbstractVector,n::Int=length(y)>>3, noverlap::Int=n>>1,window_func::Function=rect)
        N       = length(y)
        @assert N == length(t) == length(v) "y, t and v has to be the same length"
        noverlap < 0 && (noverlap = n>>1)
        W = window_func(n)
        ys = arraysplit(y,n,noverlap)
        ts = arraysplit(t,n,noverlap)
        vs = arraysplit(v,n,noverlap)
        new(y,t,v,ys,ts,vs,W)
    end
end


function Base.iterate(w::Windows3, state=1)
    state > length(w.ys) && return nothing
    ((w.ys[state],w.ts[state],w.vs[state]), state+1)
end
