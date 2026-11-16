""" source(x,a)

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/norm(x-a)
""" ∫G_kernel(x,p) = p.dA*source(x,p.x)

Monopole Green's function for a source panel `p`.
"""
∫G_kernel(x,p,args...) = p.dA*source(x,p.x)
""" ∫G_kernel(ξ,p,::QuadKernel) = ∑ᵢ wgᵢ*source(ξ,xgᵢ)

Gauss quadrature over source panel `p`.
"""
∫G_kernel(ξ,p,::QuadKernel) = (r²=sum(abs2,ξ-p.x); r²>5p.dA ? -p.dA/√r² : sum(w*source(ξ,x) for (x,w) in zip(p.xg,p.wg)))
""" ∫G_kernel(ξ,p,::PolyKernel)

Exact integrated potential over a planar polygonal panel. See Katz and Plotkin, "Low-Speed Aerodynamics" (2001)
"""
function ∫G_kernel(ξ,p,::PolyKernel)
    r = p.verts .- Ref(ξ); R = norm.(r); N = length(p.verts)
    edges = sum(1:N) do i
        m,t,j = p.inplane[i],p.tangents[i],i%N+1
        numer = R[i] + r[i]'t
        denom = R[j] + r[j]'t
        (numer > 0 && denom > 0) ? r[i]'m * log(numer/denom) : zero(r[i]'m)
    end
    Ω = primal.(ξ)==p.x ? -2π : sum(2:N-1) do i # solid angle, fan-triangulated from v₁
        a,b,c,Ra,Rb,Rc = r[1],r[i],r[i+1],R[1],R[i],R[i+1]
        2atan(a'*(b×c), Ra*Rb*Rc + (a'b)*Rc + (b'c)*Ra + (c'a)*Rb)
    end
    edges+Ω*r[1]'p.n
end

"""
    Dipole(source::GreenKernel)

Dipole panel kernel: the derivative of the `source` kernel with respect to the panel position along `p.n`.
"""
struct Dipole{K<:GreenKernel} <: GreenKernel; source::K; end
∫G_kernel(x,p,d::Dipole) = primal.(x)==p.x ? 2π*one(eltype(x)) : derivative(t->∫G_kernel(x-t*p.n,p,d.source),0) # inside limit at the centroid; moving x by -t*n ≡ moving the panel by t*n

"""
    Wake()

Kernel of a dipole strip trailing from the edge `p.verts` to x=-∞, i.e. a horseshoe vortex.
"""
struct Wake <: GreenKernel end
function ∫G_kernel(x,p,::Wake) # solid angle of the spherical triangle (a,b,-x̂)
    a,b = p.verts .- Ref(x); A,B = norm(a),norm(b)
    2atan(a'*(b×SA[-1,0,0]), A*B+a'b-b[1]*A-a[1]*B)
end

using ForwardDiff: value, partials, Dual, hessian
primal(x::Dual) = primal(value(x)); primal(x) = x # strip every Dual layer: ForwardDiff ≥1 `==` also compares partials
"""
    ∫G(x,p)

Approximate integral `∫ₚ G(x,x')da'` over panel `p`. This function enforces the jump in ∇∫G(x,x) normal to the panel.
"""
∫G(x,p) = ∫G_kernel(x,p,p.kernel)
function ∫G(d::AbstractVector{<:Dual{Tag,T,N}},p) where {Tag,T,N}
    val = ∫G_kernel(d,p,p.kernel) # use auto-diff
    primal.(d) ≠ p.x && return val
    ∂ₙ = ∂ₙself(p,p.kernel)
    ∂ = ntuple(i->∂ₙ*sum(j->partials(d[j])[i]*p.n[j],eachindex(d)),N)
    Dual{Tag}(value(val),∂...) # overwrite partials with ∇∫G(x,x)=∂ₙself*n̂ contribution
end
∂ₙself(p,kernel) = 2π # source jump
∂ₙself(p,d::Dipole) = (H = hessian(x->∫G_kernel(x,p,d.source),p.x); tr(H)-p.n'H*p.n) # ∇²G=0 ⇒ ∂ₙ²∫G=-Δₜ∫G

"""
    ∂ₙϕ(pᵢ,pⱼ;ϕ=∫G) = Aᵢⱼ

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;ϕ=∫G) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ),0)