function ridgereg(A,b,lambda, compl=false)
    n = size(A,2)
    if compl
        return real_complex_bs(A,b, lambda)
    else
        return [A; lambda*eye(n)]\[b;zeros(n)]
    end
end

function real_complex_bs(A,b, lambda=0)
    n = size(A,2)
    Ar = [real(A) imag(A)]
    if lambda > 0
        xr = [Ar; lambda*eye(2n)]\[b;zeros(2n)]
    else
        xr = Ar\b
    end
    x = complex(xr[1:n], xr[n+1:end])
end
