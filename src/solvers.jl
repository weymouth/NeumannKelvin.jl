using Krylov,LinearOperators
"""
    gmressolve!(sys, b=components(sys.panels.n,1); atol=1e-3, verbose=true)

Approximately solves a panel system using GMRES iteration to satisfy the panel
boundary conditions:
 - On body panels, the normal velocity condition `Φₙ(xᵢ,sys) = bᵢ = -U⋅n`
 - On free surface panels, Φₙ(xᵢ,sys)+sys.ℓ*Φₓₓ(xᵢ,sys) = 0`, where `ℓ=U²/g` is the
 Foude length.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `atol=1e-3`: Absolute tolerance for GMRES convergence
- `verbose=true`: Print GMRES convergence statistics

# Returns
Modified `sys` with updated panel strengths `q`

# References
Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual
algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific
and Statistical Computing, 7(3), 856-869.

# Example
```julia
BH = BarnesHut(panels)  # set-up
gmressolve!(BH)         # approximate solve
extrema(cₚ(BH))         # measure
```
"""
function gmressolve!(sys,b=components(sys.panels.n,1);atol=1e-3,verbose=true,kwargs...)
    # Make LinearOperator
    mult!(b,q) = (set_q!(sys,q); bc!(b,sys))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)

    # Solve with GMRES and return updated BarnesHutBEM
    q, stats = gmres(A, b, sys.panels.q; atol=convert(eltype(b),atol), kwargs...)
    verbose && println(stats)
    set_q!(sys,q)
end
@inline set_q!(sys,q) = (sys.panels.q .= q; sys)
@inline bc!(b::AbstractArray{T},sys,val=zero(T),h=0.3,hx=h*SA[1,0,0]) where T = AK.foreachindex(b) do i
    p = sys.panels[i]
    if p.fsbc # Φₙ - ℓ*Φₓₓ = 0
        ℓ=sys.kwargs[:ℓ]
        # b[i] = Φₙ(p,sys;val) - ℓ*(Φₓ(p.x+hx,sys;val)-Φₓ(p.x,sys;val))/h
        b[i] = Φₙ(p,sys;val) - ℓ*(Φ(p.x,sys;val)-2Φ(p.x+hx,sys;val)+Φ(p.x+2hx,sys;val))/h^2
    else      # uₙ = -Uₙ
        b[i] = Φₙ(p,sys;val)
    end
end

"""
    directsolve!(sys, b=components(sys.body.n,1))

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = bᵢ` is satisfied on body panels.

**Warning**: This function ignores sys.freesurf!

**Note**: This function is memory (and therefore time) intensive for large number of
panels N because it constructs the full N² matrix elements.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `verbose=true`: Print warning and solve time

# Returns
Modified `sys` with updated body panel strengths `q`

# Example
```julia
sys = PanelSystem(panels) # set-up
directsolve!(sys)         # solve
extrema(cₚ(sys))          # measure
```
"""
function directsolve!(sys::PanelSystem,b=components(sys.body.n,1);verbose=true)
    if verbose
        @warn "This routine ignores free surface panels and is memory intensive. See help?>directsolve!."
        @time sys.body.q .= influence(sys)\b
    else
        sys.body.q .= influence(sys)\b
    end;sys
end
function influence((;body,mirrors,kwargs)::PanelSystem)
    ϕ_sym(x,p) = sum(∫G(x .* m,p;kwargs...) for m in mirrors)
    A = Array{eltype(body.dA)}(undef,length(body),length(body))
    AK.foraxes(A,2) do j
        @simd for i in axes(A,1)
            @inbounds A[i,j] = ∂ₙϕ(body[i],body[j];ϕ=ϕ_sym)
        end
    end; A
end