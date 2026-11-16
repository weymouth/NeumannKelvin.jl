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
sys = PanelSystem(PanelTree(panels))  # set-up
gmressolve!(sys)         # approximate solve
extrema(cₚ(sys))          # measure
```
"""
function gmressolve!(sys;atol=1e-3,verbose=true,kwargs...)
    # Make LinearOperators
    b,q = rhs(sys),get_q(sys)
    mult!(b,q) = (set_q!(sys,q); bc!(b,sys))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)
    M = LinearOperator(eltype(b), length(b), length(b), false, false, (z,r)->precon!(z,sys,r))

    # Solve with GMRES and return updated PanelSystem
    q, stats = gmres(A, b, q; M, atol=convert(eltype(b),atol), kwargs...)
    verbose && println(stats)
    set_q!(sys,q)
end
# Stack the boundaries' strengths, right-hand-sides, residuals, and preconditioners
@inline slices(sys) = map((b,n)->n-length(b)+1:n,sys.boundaries,cumsum(length.(sys.boundaries)))
@inline get_q(sys) = reduce(vcat,map(b->b.q,sys.boundaries))
@inline set_q!(sys::PanelSystem,q) = (map((b,I)->set_q!(b,view(q,I)),sys.boundaries,slices(sys)); sys)
rhs(sys) = reduce(vcat,map(b->rhs(b,sys),sys.boundaries))
bc!(r,sys) = map((b,I)->bc!(view(r,I),b,sys),sys.boundaries,slices(sys))
precon!(z,sys,r) = (z .= r; map((b,I)->precon!(view(z,I),b),sys.boundaries,slices(sys)); z)
"""
    directsolve!(sys)

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = -U⋅nᵢ` is satisfied on body panels.
This function can not be used to solve a system with a `FreeSurface`.

**Note**: This function is memory (and therefore time) intensive for large number of
panels N because it constructs the full N² matrix elements. It is *not* accelerated
with a PanelTree, but *is* accelerated when Threads.nthreads()>1.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `verbose=true`: Print warning and solve time

# Returns
Modified `sys` with updated body panel strengths `q`

# Example
```julia
sys = PanelSystem(panels)     # set-up
directsolve!(sys)             # solve
extrema(cₚ(sys))               # measure
```
"""
function directsolve!(sys;verbose=true)
    q = if verbose
        @warn "This routine is memory intensive. See help?>directsolve!."
        @time influence(sys)\rhs(sys)
    else
        influence(sys)\rhs(sys)
    end; set_q!(sys,q)
end
function influence(sys)
    any(b->b isa FreeSurface,sys.boundaries) && throw(ArgumentError("directsolve! can't solve a FreeSurface"))
    I = slices(sys); N = last(last(I))
    A = zeros(eltype(first(sys.boundaries).dA),N,N)
    for (bᵢ,Iᵢ) in zip(sys.boundaries,I), (bⱼ,Iⱼ) in zip(sys.boundaries,I)
        influence!(view(A,Iᵢ,Iⱼ),bᵢ,bⱼ,sys.domain)
    end; A
end
influence!(A,bᵢ,bⱼ,domain) = (C = CartesianIndices(A); AK.foreachindex(A) do k; i,j = Tuple(C[k])
    @inbounds A[k] = condition(bᵢ,i,unit(bⱼ,j,domain))
end; A)