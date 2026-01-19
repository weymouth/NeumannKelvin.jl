module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
import NeumannKelvin: viz,viz!,AbstractPanelSystem,QuadKernel

# Default wrappers
viz(sys::AbstractPanelSystem;kwargs...) = cₚu(sys;kwargs...) |> first
function cₚu(sys;vscale=1)
    fig=Figure(); ax=Axis3(fig[1,1], aspect=:data)
    vectors = u(sys); U² = sum(abs2,sys.U); cp = @. 1-sum(abs2,vectors)/U²
    obj = viz!(ax, sys.body, cp; vectors, vscale)
    Colorbar(fig[1,2],obj;label="cₚ")
    return fig,ax
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

# Free surface
function ζviz!(fig,ax,x,y,z)
    ζmax  = maximum(abs, z); @. z[abs(z)<ζmax/20] = 0
    zeta = surface!(ax,x,y,z;shading = NoShading, colormap = :balance, colorrange = (-ζmax,ζmax))
    Colorbar(fig[1,3],zeta;label="ζ")
    return fig,ax
end
function viz(sys::FSPanelSystem)
    fig,ax = cₚu(sys)
    x,y,_ = reshape.(components(sys.freesurf.x),Ref(size(sys.fsm)))
    z = ζ(sys); z .*= sys.ℓ
    ζviz!(fig,ax,x,y,z) |> first
end
function viz(sys::NKPanelSystem,s=1/2+2π*sys.args.ℓ,h=sys.args.ℓ/3,half=true,x=-2s:h:s,y=ifelse(half,0,-s):h:s)
    fig,ax = cₚu(sys)
    z = ζ(x,y,sys); z .*= sys.args.ℓ
    ζviz!(fig,ax,x,y,z) |> first
end
end