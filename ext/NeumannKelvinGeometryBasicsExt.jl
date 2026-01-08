module NeumannKelvinGeometryBasicsExt
using NeumannKelvin,GeometryBasics
using NeumannKelvin:GreenKernel,normalize
using GeometryBasics:Mesh,Point
"""
    panelize(mesh::Mesh)-> Table{panels}

Convert a Mesh of triangles into panels
"""
NeumannKelvin.panelize(mesh::Mesh) = Table([measure(mesh.position[face]) for face in mesh.faces])
NeumannKelvin.measure(verts::AbstractArray{T}) where {T<:Point{3}} = measure(SVector{3}.(verts)...)

struct TriKernel <: GreenKernel end
"""
    measure(v₁, v₂, v₃) -> (x,n,dA,verts)

Measure the properties of a triangular panel defined by it's vertices in counter-clockwise order.
"""
function NeumannKelvin.measure(v₁, v₂, v₃)
    n = (v₂-v₁) × (v₃-v₁); dA = norm(n)/2; n = normalize(n)
    t₁ =  normalize(v₂ - v₁); t₂ = normalize(v₃-v₂); t₃ = normalize(v₁-v₃)
    (x=(v₁+v₂+v₃)/3, n=n, dA=dA, verts=SA[v₁, v₂, v₃], tangents=SA[t₁, t₂, t₃], inplane=SA[t₁×n, t₂×n, t₃×n], kernel=TriKernel())
end

""" ∫G_kernel(ξ,p,::TriKernel)

Exact integrated potential over a triangular panel. See Katz and Plotkin, "Low-Speed Aerodynamics" (2001)
"""
function NeumannKelvin.∫G_kernel(ξ, p, ::TriKernel; ignore...)
    r = p.verts .- Ref(ξ); R = norm.(r)
    edges = sum(1:3) do i
        m,t,j = p.inplane[i],p.tangents[i],i%3+1
        r[i]'m*log((R[i]+r[i]'t)/(R[j]+r[j]'t))
    end
    Ω = 2atan(r[1]'*(r[2]×r[3]),prod(R)+sum(i->r[i]'r[i%3+1]*R[(i+1)%3+1],1:3))
    @inbounds edges+r[1]'p.n*Ω
end

end