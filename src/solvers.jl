using Krylov,LinearOperators
"""
    gmressolve!(sys; verbose=true, atol=1e-3, kwargs...)

Approximately solves a panel system using GMRES iteration to satisfy the panel
boundary conditions.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `verbose=true`: Print GMRES convergence statistics
- `atol=1e-3, kwargs...`: Absolute tolerance and other arguments passed to gmres

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
function gmressolve!(sys;atol=1e-3,verbose=true,kwargs...)
    # Make LinearOperators
    b = rhs(sys.panels,sys.U)
    mult!(b,q) = (set_q!(sys,q); bodybc!(b,sys))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)
    M = LinearOperator(eltype(b), length(b), length(b), false, false, (z,r)->precon!(z,sys,r))

    # Solve with GMRES and return updated BarnesHutBEM
    q, stats = gmres(A, b, sys.panels.q; M, atol=convert(eltype(b),atol), kwargs...)
    verbose && println(stats)
    set_q!(sys,q)
end
@inline rhs(panels,U) = -sum(components(panels.n) .* U)
@inline bodybc!(b,sys) = AK.foreachindex(i-> b[i] = Φₙ(sys.panels[i],sys),b) # body: Φₙ = -Uₙ
@inline precon!(z,sys,r) = z .= r # identity
"""
    directsolve!(sys)

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = -U⋅nᵢ` is satisfied on body panels.

*Note*: This function is memory (and therefore time) intensive for large number of
panels N because it constructs the full N² matrix elements.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `verbose=true`: Print warning and solve time

# Returns
Modified `sys` with updated body panel strengths `q`

# Example
```julia
sys = BodyPanelSystem(panels) # set-up
directsolve!(sys)             # solve
extrema(cₚ(sys))               # measure
```
"""
function directsolve!(sys;verbose=true)
    if verbose
        @warn "This routine ignores free surface panels and is memory intensive. See help?>directsolve!."
        @time sys.panels.q .= influence(sys)\rhs(sys.panels,sys.U)
    else
        sys.panels.q .= influence(sys)\rhs(sys.panels,sys.U)
    end;sys
end
function influence((;panels,mirrors)::AbstractPanelSystem)
    ϕ_sym(x,p) = sum(m->∫G(x .* m,p),mirrors)
    A = Array{eltype(panels.dA)}(undef,length(panels),length(panels))
    AK.foraxes(A,2) do j
        @simd for i in axes(A,1)
            @inbounds A[i,j] = ∂ₙϕ(panels[i],panels[j];ϕ=ϕ_sym)
        end
    end; A
end