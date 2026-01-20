module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
import NeumannKelvin: viz,cₚu,xyζ,QuadKernel

# Table and free surface viz functions
function viz(panels::Union{Table,PanelTree},values=panels.dA; vectors=panels.n, label="", vscale=1, kwargs...)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    viz!(fig,ax,panels,values;vectors,vscale,label,kwargs...)
    return fig
end
function viz(sys::Union{FSPanelSystem,NKPanelSystem};kwargs...)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    cp,vectors = cₚu(sys); viz!(fig,ax,sys.body,cp;vectors,label="cₚ",kwargs...)
    x,y,z = xyζ(sys)
    ζmax  = maximum(abs, z); @. z[abs(z)<ζmax/20] = 0
    zeta = surface!(ax,x,y,z;shading = NoShading, colormap = :balance, colorrange = (-ζmax,ζmax))
    Colorbar(fig[1,3],zeta;label="ζ")
    return fig
end

# Generate full mesh and color array
function viz!(fig::Figure,ax::Axis3, panels::Union{Table,PanelTree}, values=panels.dA; vectors=panels.n, 
    vscale=1, label=nothing, clims=extrema(values), kwargs...)

    # Mesh geometry (vertices, normals, and colors)
    obj = mesh!(ax, panelmesh.(panels,panels.kernel); color=values, colorrange=clims, shading=NoShading, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...;
        lengthscale=0.025vscale, color=values, colorrange=clims, kwargs...)

    # Colorbar
    Colorbar(fig[1,2],obj;label,kwargs...)
end
const quadface = decompose(QuadFace{GLIndex},Tessellation(Rect(0, 0, 1, 1), (2, 2)))
panelmesh(p,::QuadKernel) = GeometryBasics.Mesh(Point3f.(p.verts), quadface, normal=Point3f.(p.nverts))
panelmesh(p,ignore...) = GeometryBasics.Mesh(Point3f.(p.verts), [TriangleFace{GLIndex}(1,2,3)])
end