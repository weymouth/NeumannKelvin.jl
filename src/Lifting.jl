"""
    Lifting(panels, te)

Lifting body boundary of dipole `panels` with a flat wake trailing to x=-∞ from each trailing edge
`(v₁,v₂)` in `te`. The wake strength is set by the linear Kutta condition, and the total potential
inside the body is zero. The flow must be `U=[-|U|,0,0]`, so rotate the body to set the angle of attack.

**Note**: The body must be closed with neighbouring panels sharing vertices (`panelize` doesn't produce
such grids), the panels are flattened to polygons, and each trailing edge must be a full edge of exactly two panels.
"""
struct Lifting{T,P<:Table,W<:Table,C,L} <: PanelWrapper{T}
    panels::P # dipole panels
    wake::W   # trailing strips, with the indices `u`,`l` of the panels above & below
    cuts::C   # trailing edges
    loops::L  # surface velocity weights
    Lifting(panels::P,wake::W,cuts::C,loops::L) where {P,W,C,L} = new{eltype(panels),P,W,C,L}(panels,wake,cuts,loops)
end
function Lifting(body,te)
    panels = Table(map(p->merge(measure(p.verts...),(;kernel=Dipole(PolyKernel()),q=zero(p.dA))),body))
    Lifting(panels,Table(map(e->trailing(panels,e),te)),te,nothing)
end
function trailing(panels,(v₁,v₂))
    ij = findall(p->any(e->along(e,(v₁,v₂))&&along((v₁,v₂),e),edges(p)),panels)
    length(ij)==2 || throw(ArgumentError("trailing edge $v₁→$v₂ must be a full edge of exactly two panels"))
    n = normalize((v₂-v₁)×SA[-1,0,0]); u,l = panels.n[ij[1]]'n>0 ? ij : reverse(ij)
    (;x=(v₁+v₂)/2, n, verts=SA[v₁,v₂], kernel=Wake(), q=zero(n[1]), u, l)
end
setup(L::Lifting,domain) = Lifting(L.panels,L.wake,L.cuts,loopweights(L.panels,L.cuts,domain.mirrors))
Base.show(io::IO,L::Lifting) = println(io,"Lifting($(length(L.panels)) panels, $(length(L.wake)) trailing edges)")

check(::Lifting,sys) = checkU(sys)
@inline set_q!(L::Lifting,q) = (L.panels.q .= q; L.wake.q .= view(q,L.wake.u) .- view(q,L.wake.l); L)
@inline Φ_boundary(x,L::Lifting,d) = Φ_boundary(x,L.panels,d)+Φ_boundary(x,L.wake,d)
rhs(L::Lifting,sys) = map(x->-sys.U'x,L.x)
condition(L::Lifting,i,f) = f(L.x[i])
influence!(A,bᵢ,L::Lifting,domain) = (influence!(A,bᵢ,L.panels,domain);
    A .+= influence!(similar(A,size(A,1),length(L.wake)),bᵢ,L.wake,domain)*kutta(L))
kutta(L) = (K = zeros(length(L.wake),length(L)); foreach(((k,w),)->(K[k,w.u] = 1; K[k,w.l] = -1),enumerate(L.wake)); K)
u(L::Lifting,sys) = map(w->-4π*sum(((j,wⱼ),)->L.q[j]*wⱼ,w),L.loops)
cₚ(L::Lifting,sys) = 1 .- sum.(abs2,u(L,sys)) ./ sum(abs2,sys.U)