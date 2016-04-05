using JuMP


N = 100
f0 = 3
fs = 10

t   = 0:1/fs:((N-1)/fs)
y   = sin(2π*f0*t)
f = 0.1:0.1:4

N = length(y)
Nf = length(f)
A = zeros(Complex128,N,Nf)
for n = 1:N, fn=1:Nf
    A[n,fn] = exp(-2*pi*1im*f[fn]*t[n])
end



m = Model()

@defVar(m,Xv[1:Nf],:Complex)
@defVar(m,ΔBv[1:N])
@defVar(m,ΔAv[1:N,1:Nf])

for i = 1:N
    @addConstraint(m, ((A[i,:] + ΔAv[i,:])*Xv)[1]== y[i] + ΔBv[i])
end
setObjective(m, :Min, sum([10*ΔAv ΔBv].^2))

# print(m)

status = solve(m)

println("Objective value: ", getObjectiveValue(m))
X   = getValue(Xv)
# ΔA  = getValue(ΔA)
# B   = getValue(ΔB)
println("X = ", X)
println("ΔA = ", ΔA)
println("ΔB = ", ΔB)

mΔA = mean(ΔA,1)

Δf = -log(ΔA./y)./(1im*2π*t)

# OK, säg att man vill hitta den bästa approximationen av en signal, som en dekomposition av komplexa sinusar. Man väljer en sparse setup av frekvenser, men man är inte helt säker på att man har valt bra värden. Därför kör man TLS, för att tillåta lite pertubation i A-matrisen också. Eftersom man vill ha samma pertubation för alla datapunkter räknar man ut medelpertubationen för varje frekvens, uppdaterar sin frekvensvektor och gör om allting igen. Konvergerar detta mot någonting? I så fall bör det konvergera mot en nice uppsättning frekvenser som klarar av att beskriva datan bäst.
