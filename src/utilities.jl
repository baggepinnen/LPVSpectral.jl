"""`ridgereg(A,b,λ)`
Accepts `λ` to solve the ridge regression problem using the formulation `[A;λI]\\[b;0]`"""
function ridgereg(A,b,λ)
    n = size(A,2)
    [A; λ*eye(n)]\[b;zeros(n)]
end

"""`real_complex_bs(A,b, λ=0)`
Replaces the backslash operator for complex arguments. Expands the A-matrix into `[real(A) imag(A)]` and performs the computation using real arithmetics. Optionally accepts `λ` to solve the ridge regression problem using the formulation `[A;λI]\\[b;0]`"""
function real_complex_bs(A,b, λ=0)
    n = size(A,2)
    Ar = [real(A) imag(A)]
    if λ > 0
        xr = [Ar; λ*eye(2n)]\[b;zeros(2n)]
    else
        xr = Ar\b
    end
    x = complex(xr[1:n], xr[n+1:end])
end
