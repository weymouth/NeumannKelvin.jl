module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
import NeumannKelvin: viz,viz!,AbstractPanelSystem,QuadKernel

# Default wrappers
function viz(sys::BodyPanelSystem;vscale=1)
    v = u(sys); U² = sum(abs2,sys.U); cp = @. 1-sum(abs2,v)/U²
    viz(sys.body, cp; vectors=v, vscale, label="cₚ")
end
function viz(sys::FSPanelSystem)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    v = u(sys); U² = sum(abs2,sys.U); cp = @. 1-sum(abs2,v)/U²
    cpp = viz!(ax,sys.body, cp; vectors=v)
    Colorbar(fig[1,2],cpp;label="cₚ")
    zeta = viz!(ax,sys,Val(ζ))
    Colorbar(fig[1,3],zeta;label="ζ")
    return fig
end
function viz(panels::Union{Table,PanelTree},values=panels.dA; vectors=collect(panels.n), vscale = 1, kwargs...)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    obj = viz!(ax,panels,values;vectors,vscale,kwargs...)
    Colorbar(fig[1,2],obj;kwargs...)
    return fig
end

# Generate full mesh and color array
function viz!(ax::Axis3, panels::Union{Table,PanelTree}, values=panels.dA; vectors=collect(panels.n), vscale = 1,
    clims = extrema(values), kwargs...)

    # Mesh geometry (vertices, normals, and colors)
    obj = mesh!(ax, panelmesh.(panels,panels.kernel); color=values, colorrange=clims, shading=NoShading, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...; 
        lengthscale=0.025vscale, color=values, colorrange=clims, kwargs...)
    obj
end
const quadface = decompose(QuadFace{GLIndex},Tessellation(Rect(0, 0, 1, 1), (2, 2)))
panelmesh(p,::QuadKernel) = GeometryBasics.Mesh(Point3f.(p.verts), quadface, normal=Point3f.(p.nverts))
panelmesh(p,ignore...) = GeometryBasics.Mesh(Point3f.(p.verts), [TriangleFace{GLIndex}(1,2,3)])

# Free surface plot
function viz!(ax::Axis3,sys,::Val{ζ}; kwargs...)
    z = ζ(sys)*sys.ℓ # unscaled elevation
    x,y,_ = reshape.(components(sys.freesurf.x),Ref(size(z)))
    ζmax  = maximum(abs, z); @. z[abs(z)<ζmax/20] = 0
    surface!(ax,x,y,z;shading = NoShading, colormap = :balance, colorrange = (-ζmax,ζmax), kwargs...)
end
end