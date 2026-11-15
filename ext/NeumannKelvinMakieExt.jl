module NeumannKelvinMakieExt
using NeumannKelvin,Makie
import NeumannKelvin: viz,viz!,xyζ,bodies,QuadKernel,PanelWrapper

# viz wrappers
function viz(panels::Union{Table,PanelWrapper},values=panels.dA; vectors=panels.n, title="", kwargs...)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data; title)
    viz!(fig,ax,panels,values;vectors,kwargs...)
    return fig
end
function viz(sys::PanelSystem; freesurf=true, title="", fsargs=NamedTuple(), kwargs...)
    # body plots
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data; title)
    for (k,b) in enumerate(bodies(sys))
        viz!(fig,ax,b,cₚ(b,sys);vectors=u(b,sys),label="cₚ",colorbar_position=(1,1+k),kwargs...)
    end
    fs = xyζ(sys;fsargs...)
    (isnothing(fs) || !freesurf) && return fig
    # add free surface
    x,y,z = fs
    ζmax  = maximum(abs, z); @. z[abs(z)<ζmax/20] = 0
    zeta = surface!(ax,x,y,z;shading = NoShading, colormap = :balance, colorrange = (-ζmax,ζmax))
    Colorbar(fig[1,2+length(bodies(sys))],zeta;label="ζ")
    return fig
end

# Generate mesh from Table data
function viz!(fig::Figure,ax::Axis3, panels::Union{Table,PanelWrapper}, values; vectors=nothing, 
    vscale=1, label="", colormap=:viridis, colorrange=extrema(values), colorbar_position=(1,2), kwargs...)

    # Mesh geometry (vertices, normals, and colors)
    obj = mesh!(ax, panelmesh.(panels,panels.kernel); color=values, colorrange, colormap, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...;
        lengthscale=0.1vscale, color=values, colorrange, colormap, kwargs...)

    # Colorbar
    Colorbar(fig[colorbar_position...],obj;label,kwargs...)
end
using Makie.GeometryBasics
const quadface = decompose(QuadFace{GLIndex},Tessellation(Rect(0, 0, 1, 1), (2, 2)))
panelmesh(p,::QuadKernel) = GeometryBasics.Mesh(Point3f.(p.verts), quadface, normal=Vec3f.(p.nverts))
panelmesh(p,ignore...) = GeometryBasics.Mesh(Point3f.(p.verts), [TriangleFace{GLIndex}(1,i,i+1) for i in 2:length(p.verts)-1])
end