module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
# Generate full triangle mesh and color array
function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=panels.n, clims = extrema(values), kwargs...)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=:data)

    # Mesh geometry (vertices, normals, and colors)
    mesh!(ax, panelmesh.(panels,panels.kernel); color=values, colorrange=clims, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...; 
        lengthscale=1, color=values, colorrange=clims, kwargs...)
    Colorbar(fig[1, 2];colorrange=clims, kwargs...)
    fig
end
const quadface = decompose(QuadFace{GLIndex},Tessellation(Rect(0, 0, 1, 1), (2, 2)))
panelmesh(p,::NeumannKelvin.QuadKernel) = GeometryBasics.Mesh(Point3f.(p.verts), quadface, normal=Point3f.(p.nverts))
panelmesh(p,ignore...) = GeometryBasics.Mesh(Point3f.(p.verts), [TriangleFace{GLIndex}(1,2,3)])
end