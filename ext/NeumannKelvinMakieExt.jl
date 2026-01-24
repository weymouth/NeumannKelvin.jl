module NeumannKelvinMakieExt
using NeumannKelvin,Makie
import NeumannKelvin: viz,cₚu,xyζ,QuadKernel,AbstractPanelSystem

# viz wrappers
function viz(panels::Union{Table,PanelTree},values=panels.dA; vectors=panels.n, title="", kwargs...)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data; title)
    viz!(fig,ax,panels,values;vectors,kwargs...)
    return fig
end
function viz(sys::AbstractPanelSystem; freesurf=true, title="", fsargs=NamedTuple(), kwargs...)
    # body plot
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data; title)
    cp,vectors = cₚu(sys)
    viz!(fig,ax,sys.body,cp;vectors,label="cₚ",kwargs...)
    (typeof(sys)<:BodyPanelSystem || !freesurf) && return fig
    # add free surface
    x,y,z = xyζ(sys;fsargs...)
    ζmax  = maximum(abs, z); @. z[abs(z)<ζmax/20] = 0
    zeta = surface!(ax,x,y,z;shading = NoShading, colormap = :balance, colorrange = (-ζmax,ζmax))
    Colorbar(fig[1,3],zeta;label="ζ")
    return fig
end

# Generate mesh from Table data
function viz!(fig::Figure,ax::Axis3, panels::Union{Table,PanelTree}, values; vectors=nothing, 
    vscale=1, label="", colorrange=extrema(values), kwargs...)

    # Mesh geometry (vertices, normals, and colors)
    obj = mesh!(ax, panelmesh.(panels,panels.kernel); color=values, colorrange, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...;
        lengthscale=0.1vscale, color=values, colorrange, kwargs...)

    # Colorbar
    Colorbar(fig[1,2],obj;label,kwargs...)
end
using Makie.GeometryBasics
const quadface = decompose(QuadFace{GLIndex},Tessellation(Rect(0, 0, 1, 1), (2, 2)))
panelmesh(p,::QuadKernel) = GeometryBasics.Mesh(Point3f.(p.verts), quadface, normal=Point3f.(p.nverts))
panelmesh(p,ignore...) = GeometryBasics.Mesh(Point3f.(p.verts), [TriangleFace{GLIndex}(1,2,3)])
end