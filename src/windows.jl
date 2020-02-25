abstract type AbstractWindows end

# Window2 ==========================================================
struct Windows2 <: AbstractWindows
    y::AbstractVector
    t::AbstractVector
    ys
    ts
    W
    """
        Windows2(y, t, n=length(y)÷8, noverlap=-1, window_func=identity)

    noverlap = -1 sets the noverlap to n/2

    #Arguments:
    - `y`: Signal
    - `t`: Time vector
    - `n`: Number of datapoints per window
    - `noverlap`: Overlap between windows
    - `window_func`: Function to apply over window
    """
    function Windows2(y,t,n::Int=length(y)>>3, noverlap::Int=n>>1,window_func=identity)

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
    state > length(w.ys) && return nothing
    ((w.ys[state],w.ts[state]), state+1)
end



function Base.merge(yf::AbstractVector{<:AbstractVector},w::Windows2)
    ym = zeros(eltype(yf[1]), length(w.y))
    counts = zeros(Int, length(w.y))
    dpw = length(w.ys[1])
    inds =  1:dpw

    for i in 1:w.nw
        ym[inds] .+= yf[i]
        counts[inds] .+= 1
        inds = inds .+ (w.dpw-w.noverlap)
        inds =  inds[1]:min(inds[end], length(ym))
    end
    ym ./= max.(counts,1)
end

# Window3 ==========================================================
"""
    Windows3(y, t, v, n=length(y)÷8, noverlap=-1, window_func=identity)

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
    function Windows3(y::AbstractVector,t::AbstractVector,v::AbstractVector,n::Int=length(y)>>3, noverlap::Int=n>>1,window_func::Function=identity)
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

Base.length(w::AbstractWindows) = length(w.ys);
