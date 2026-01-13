module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
import NeumannKelvin: viz,viz!,AbstractPanelSystem,QuadKernel

# Default wrappers
function viz(sys::AbstractPanelSystem)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    cp = viz!(ax,sys.body, cₚ(sys); vectors=nothing)
    Colorbar(fig[1,2],cp;label="cₚ")
    isnothing(sys.freesurf) && return fig
    zeta = viz!(ax,sys,Val(ζ))
    Colorbar(fig[1,3],zeta;label="ζ")
    return fig
end
function viz(t::Union{Table,AbstractPanelSystem},args...;kwargs...)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    obj = viz!(ax,t,args...;kwargs...)
    Colorbar(fig[1,2],obj;kwargs...)
    return fig
end

# Generate full mesh and color array
function viz!(ax::Axis3, panels::Table, values=panels.dA; vectors=panels.n, 
    clims = extrema(values), kwargs...)

    # Mesh geometry (vertices, normals, and colors)
    obj = mesh!(ax, panelmesh.(panels,panels.kernel); color=values, colorrange=clims, shading=NoShading, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...; 
        lengthscale=0.025, color=values, colorrange=clims, kwargs...)
    obj
end
const quadface = decompose(QuadFace{GLIndex},Tessellation(Rect(0, 0, 1, 1), (2, 2)))
panelmesh(p,::QuadKernel) = GeometryBasics.Mesh(Point3f.(p.verts), quadface, normal=Point3f.(p.nverts))
panelmesh(p,ignore...) = GeometryBasics.Mesh(Point3f.(p.verts), [TriangleFace{GLIndex}(1,2,3)])

# Free surface plot
function viz!(ax::Axis3,sys,::Val{ζ}; kwargs...)
    x,y,_ = reshape.(components(sys.freesurf.x),Ref(sys.fssize))
    z = reshape(ζ(sys)*sys.ℓ[1],sys.fssize)
    ζmax  = maximum(abs, z); @. z[abs(z)<ζmax/20] = 0
    surface!(ax,x,y,z;shading = NoShading, colormap = :balance, kwargs...)
end
end