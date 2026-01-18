"""
    FSPanelSystem(body,freesurf::Matrix{QuadPanels}; Umag=1, ℓ=1, sym_axes=(), wrap=PanelTree)

A PanelSystem defined by `body` and `freesurf` source panels.

**Note**: `freesurf` must be a matrix of quadralateral panels with the first index
of the matrix aligned with -x. i.e. `freesurf[i+1,j].x-freesurf[i,j].x ≈ [-Δxᵢⱼ,0,0]`.
Similarly, the direction of the flow will always be `Û=-x̂`. Relative flow vectoring
(i.e. drift angle) must be acheived by rotating the body.

keyword arguments:
- `Umag` sets the *magnitude* of the background flow
- `ℓ=Umag²/g` sets the Froude length
- `sym_axes, wrap` see BodyPanelSystem

# Usage
```julia
sys = FSPanelSystem(body_panels,freesurf;ℓ=1/2π,wrap=PanelTree) # body + free surface
gmressolve!(sys, atol=1e-6)  # approximate solve
extrema(cₚ(sys))             # check solution quality
```
"""
struct FSPanelSystem{B,F,D,L,T,M} <: AbstractPanelSystem
    body::B      # body panels
    freesurf::F  # free surface panels
    fsm::D       # free-surface sized Matrix
    ℓ::L         # Froude-length
    U::SVector{3,T}
    mirrors::M
end
function FSPanelSystem(body,freesurf::AbstractMatrix; Umag=1, ℓ=1, sym_axes=(), wrap=PanelTree, kwargs...)
    # Lots of sanitary input checks...
    fsm = zeros(eltype(body.dA),size(freesurf)); freesurf = Table(freesurf)
    eltype(freesurf.dA) != eltype(fsm) && throw(ArgumentError("Floating point type of body and freesurf panels must match"))
    freesurf.n[1][3]>1 && throw(ArgumentError("`freesurf` panels must point down."))
    dx = freesurf.x[2]-freesurf.x[1]
    (dx[1]>-abs(dx[2]) || dx[3]≠0) && throw(ArgumentError("first index of `freesurf` must align with -x."))

    FSPanelSystem(snug(body,wrap;kwargs...), snug(freesurf,wrap;kwargs...), fsm, ℓ, SA[-abs(Umag),0,0], mirrors(sym_axes...))
end

# Pretty printing
Base.show(io::IO, sys::FSPanelSystem) = println(io, "FSPanelSystem($(length.(domains(sys))) panels, ℓ=$(sys.ℓ)")
function Base.show(io::IO, ::MIME"text/plain", sys::FSPanelSystem)
    println(io,"FSPanelSystem")
    print(  io, "  freesurf: "); show(io,sys.freesurf)
    println(io, "     size: $(size(sys.fsm))")
    println(io, "     panel type: $(eltype(sys.freesurf.kernel))")
    println(io, "  Froude length ℓ: $(sys.ℓ)")
    abstract_show(io,sys)
end

# Calculate the potential
@inline domains(sys) = (sys.body,sys.freesurf)
Φ(x,sys::FSPanelSystem) = sum(m->Φ_dom(x .* m,sys.body)+Φ_dom(x .* m,sys.freesurf),sys.mirrors)

# Set/Get the strength
@inline dviews(q,sys) = ((Nb,Nfs) = length.(domains(sys));view.(Ref(q),(1:Nb,Nb+1:Nb+Nfs)))
@inline set_q!(sys::FSPanelSystem,q) = (set_q!.(domains(sys),dviews(q,sys)); sys)
@inline get_q(sys::FSPanelSystem) = [sys.body.q; sys.freesurf.q]

# Set the rhs, solution, fsbc and preconditioner
rhs(sys::FSPanelSystem) = [rhs(sys.body,sys.U); zeros_like(sys.freesurf.dA)]
function bc!(b,sys::FSPanelSystem)
    p = sys.fsm; Nᵢ = size(p,1); Nb = length(sys.body); ℓ = sys.ℓ
    # body: Φₙ = -Uₙ
    AK.foreachindex(i-> b[i] = Φₙ(sys.body[i],sys),view(b,1:Nb))
    # freesurf: Φₙ-ℓΦₓₓ=0
    AK.foreachindex(i-> p[i] = Φ(sys.freesurf.x[i],sys), p) # fill p->Φ
    AK.foreachindex(p) do i
        b[i+Nb] = Φₙ(sys.freesurf[i],sys)
        (i-1)%Nᵢ<3 && return
        h = extent(view(sys.freesurf.x,i-1:i))[1]
        @inbounds b[i+Nb] -= ℓ*(2p[i]-5p[i-1]+4p[i-2]-p[i-3])/h^2 # 2nd order upwind FD
    end
end
function precon!(z,sys::FSPanelSystem,r)
    Nᵢ,Nⱼ = size(sys.fsm); Nb = length(sys.body)
    z .= r # identity preconditioner
    # sweep resdual information downstream
    AK.foreachindex(view(z,1:Nⱼ)) do j
    for i in (2:Nᵢ) .+ (Nb+(j-1)*Nᵢ)
        @inbounds z[i] += z[i-1]
    end;end
end

"""
    ζ([x::SVector{3},] sys)

Scaled linear free surface elevation `ζ/ℓ=Φₓ/|U|` induced by **solved** panel system `sys`.
If no location `x` is given, a vector of ζ at all freesurf centers is returned.

See also: [`Φ`](@ref)
"""
ζ(x::SVector{3},sys) = derivative(t->Φ(x+t*SA[1,0,0],sys),0)/norm(sys.U)
function ζ(sys::FSPanelSystem)
    b = sys.fsm
    AK.foreachindex(b) do i
        b[i] = ζ(sys.freesurf.x[i],sys)
    end; b
end
