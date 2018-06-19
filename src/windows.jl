# Window2 ==========================================================
"""Windows2(y,t,nw,noverlap,window_func)
noverlap = -1 sets the noverlap to dpw/2
"""
struct Windows2
    y::AbstractVector
    t::AbstractVector
    nw::Int
    dpw::Int
    noverlap::Int
    W
    function Windows2(y,t,nw,noverlap,window_func)
        N   = length(y)
        @assert N == length(t) "y and t has to be the same length"
        dpw = floor(Int64,N/nw)
        if noverlap == -1
            noverlap = round(Int64,dpw/2)
        end
        W = window_func(dpw)
        Windows2(y,t,nw,dpw,noverlap,W)
    end
end


Base.start(::Windows2) = 0

function Base.next(w::Windows2, state)
    if state < w.nw-1
        inds =  (1:w.dpw)+state*(w.dpw-w.noverlap)
    else
        inds =  state*(w.dpw-w.noverlap):length(w.y)
    end
    ((w.y[inds],w.t[inds]), state+1)
end

Base.done(w::Windows2, state) = state >= w.nw;

Base.length(w::Windows2) = w.nw;


# Window3 ==========================================================
"""Windows3(y,t,v,nw::Integer,noverlap::Integer,window_func::Function)
noverlap = -1 sets the noverlap to dpw/2
"""
struct Windows3
    y::AbstractVector
    t::AbstractVector
    v::AbstractVector
    nw::Int
    dpw::Int
    noverlap::Int
    W
end

function Windows3(y,t,v,nw::Integer,noverlap::Integer,window_func::Function)
    N       = length(y)
    @assert N == length(t) == length(v) "y, t and v has to be the same length"
    dpw     = floor(Int64,N/nw)
    if noverlap == -1
        noverlap = round(Int64,dpw/2)
    end
    W       = window_func(dpw)
    Windows3(y,t,v,nw,dpw,noverlap,W)
end

Base.start(::Windows3) = 0

function Base.next(w::Windows3, state)
    if state < w.nw
        inds =  (1:w.dpw)+state*(w.dpw-w.noverlap)
    else
        inds =  state*(w.dpw-w.noverlap):length(w.y)
    end
    ((w.y[inds],w.t[inds],w.v[inds]), state+1)
end

Base.done(w::Windows3, state) = state >= w.nw;

Base.length(w::Windows3) = w.nw;
