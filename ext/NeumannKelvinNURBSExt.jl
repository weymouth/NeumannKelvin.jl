module NeumannKelvinNURBSExt
using NeumannKelvin,NURBS
using StaticArrays

# Define a gNURBS struct that can be called as a function to evaluate the NURBS surface at given parameters (u,v)
struct gNURBS{du,dv,A<:AbstractMatrix,U<:AbstractVector,V<:AbstractVector,W<:AbstractMatrix} <: Function
    pnts::A
    uknots::U
    vknots::V
    wgts::W
    function gNURBS(pnts,uknots,vknots,wgts)
        size(wgts) == size(pnts) || throw(ArgumentError("Size of control points must match size of weights"))
        cu,cv = size(pnts)
        du = length(uknots)-cu-1; du<1 && throw(ArgumentError("Degree in u direction must be at least 1"))
        dv = length(vknots)-cv-1; dv<1 && throw(ArgumentError("Degree in v direction must be at least 1"))
        new{du,dv,typeof(pnts),typeof(uknots),typeof(vknots),typeof(wgts)}(pnts,uknots,vknots,wgts)
    end
end
function (s::gNURBS{du,dv})(u::Tu,v::Tv) where {du,dv,Tu,Tv} 
    T = promote_type(Tu,Tv,eltype(eltype(s.pnts)))
    psum,wsum = zero(SVector{3,T}),zero(T)
    cu,cv = size(s.pnts)
    
    u₀ = searchsortedlast(view(s.uknots, du+1:cu), u)
    v₀ = searchsortedlast(view(s.vknots, dv+1:cv), v)
    uN = basis_funs(s.uknots, T(u), u₀+du, Val(du))
    vN = basis_funs(s.vknots, T(v), v₀+dv, Val(dv))
    
    # Combine basis functions with control points & weights
    @inbounds for i in 0:du, j in 0:dv
        prod = uN[i+1] * vN[j+1] * s.wgts[u₀ + i, v₀ + j]
        psum += prod * s.pnts[u₀ + i, v₀ + j]
        wsum += prod
    end    
    return psum / wsum
end    

# Iteratively compute basis functions (Algorithm A2.2 from NURBS book)
@inline function basis_funs(knots, u::T, span, ::Val{degree}) where {T,degree}
    N = @MVector zeros(T, degree+1)
    left = @MVector zeros(T, degree+1)
    right = @MVector zeros(T, degree+1)
    
    N[1] = one(T)
    @inbounds for j in 1:degree
        left[j+1] = u - knots[span+1-j]
        right[j+1] = knots[span+j] - u
        saved = zero(T)
        for r in 0:j-1
            temp = N[r+1] / (right[r+2] + left[j-r+1])
            N[r+1] = saved + right[r+2] * temp
            saved = left[j-r+1] * temp
        end
        N[j+1] = saved
    end
    return SVector(N)
end

# Define a wrapper for NURBSsurface and use it in exported panel functions
gNURBS(patch::NURBSsurface) = gNURBS(patch.controlPoints, patch.uBasis.knotVec, patch.vBasis.knotVec, patch.weights)
NeumannKelvin.panelize(patches::AbstractArray{T},args...;kwargs...) where {T<: NURBSsurface} = mapreduce(patch->panelize(patch,args...;kwargs...),vcat,patches)
NeumannKelvin.panelize(patch::NURBSsurface,args...;kwargs...) = panelize(gNURBS(patch),args...;kwargs...)
NeumannKelvin.measure(patch::NURBSsurface,args...;kwargs...) = measure(gNURBS(patch),args...;kwargs...)
end