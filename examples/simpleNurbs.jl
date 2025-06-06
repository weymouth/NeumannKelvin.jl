using StaticArrays

struct NurbsCurve{n,l,k} <: Function
    pnts ::SMatrix{n,l}
    knots::SVector{k}
    wgts ::SVector{l}
end
function BSpline(pnts::SMatrix{n,count,T};degree=1) where {n,count,T<:AbstractFloat}
    @assert degree <= count - 1 "Invalid B-Spline: the degree should be less than the number of control points minus 1."
    knots = SA{T}[[zeros(degree); collect(range(0, count-degree) / (count-degree)); ones(degree)]...]
    weights = SA{T}[ones(count)...]
    NurbsCurve(pnts,knots,weights)
end

function (r::NurbsCurve{n,l,k})(u::T) where {T,n,l,k}
    pt = zeros(SVector{n,T}); wsum=zero(T)
    degree = k-l-1; iend = findint(u,r.knots,l)
    @inbounds @simd for i in iend-degree:iend
        prod = Bd(r.knots,u,i,Val(degree))*r.wgts[i]
        pt += prod*r.pnts[:,i]; wsum += prod
    end
    pt/wsum
end
findint(u,knots,l)::Int = (ff = findfirst(>(u), @view knots[1:l]); isnothing(ff) ? l : ff-1)

"""
    Bd(knot, u, k, ::Val{d}) where d

Compute the Cox-De Boor recursion for B-spline basis functions.
- `knot` : A Vector containing the knots of the B-Spline, with the knot value `k ∈ [0,1]`.
- `u` : A Float representing the value of the parameter on the curve at which the basis function is computed, `u ∈ [0,1]`
- `k` : An Integer representing which basis function is computed.
- `d` : An Integer representing the order of the basis function to be computed.
"""
@inline Bd(knots, u, k, ::Val{0}) = Int(knots[k]≤u<knots[k+1] || knots[k+1]==1≤u)
function Bd(knots, u::T, k, ::Val{d}) where {T,d}
    ((u-knots[k])/max(eps(T),knots[k+d]-knots[k])*Bd(knots,u,k,Val(d-1))
    +(knots[k+d+1]-u)/max(eps(T),knots[k+d+1]-knots[k+1])*Bd(knots,u,k+1,Val(d-1)))
end

struct NurbsSurface{n,lu,lv,ku,kv} <: Function
    pnts   ::SMatrix{lu,lv}
    knotsu ::SVector{ku}
    knotsv ::SVector{kv}
    wgts   ::SMatrix{lu,lv}
end
using NURBS
function NurbsSurface(s::NURBSsurface,scale=1f0,shift=SA[0,0,0])
    NURBS.scale!(s,scale); NURBS.translate!(s,shift)
    pnts = s.controlPoints; lu,lv = size(pnts); n = length(first(pnts))
    knotsu = s.uBasis.knotVec; ku = length(knotsu)
    knotsv = s.vBasis.knotVec; kv = length(knotsv)
    wghts = s.weights
    NurbsSurface{n,lu,lv,ku,kv}(SMatrix{lu,lv}(pnts),SVector{ku}(knotsu),SVector{kv}(knotsv),SMatrix{lu,lv}(wghts))
end

function (s::NurbsSurface{n,lu,lv,ku,kv})(u::Tu,v::Tv) where {Tu,Tv,n,lu,lv,ku,kv}
    T = promote_type(Tu,Tv)
    pt = zeros(SVector{n,T}); wsum=zero(T)
    degreeu = ku-lu-1; iend = findint(u,s.knotsu,lu)
    degreev = kv-lv-1; jend = findint(v,s.knotsv,lv)
    @inbounds @simd for i in iend-degreeu:iend; @simd for j in jend-degreev:jend
        Bu = Bd(s.knotsu,u,i,Val(degreeu))
        Bv = Bd(s.knotsv,v,j,Val(degreev))
        prod = Bu*Bv*s.wgts[i,j]
        pt += prod*s.pnts[i,j]; wsum += prod
    end;end
    pt/wsum
end
