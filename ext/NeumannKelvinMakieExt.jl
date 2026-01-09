module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
# Generate full triangle mesh and color array
function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=0.3panels.n, clims = extrema(values), kwargs...)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=:data)

    # Mesh geometry (vertices, normals, and colors)
    face = decompose(QuadFace{GLIndex},Tessellation(Rect(0,0,1,1),(2,2))) # 2x2 quad type
    quad(panel) = GeometryBasics.Mesh(Point3f.(panel.verts), face, normal=Point3f.(panel.nverts))  
    mesh!(ax, quad.(panels); color=values, colorrange=clims, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...; 
        lengthscale=0.5, color=values, colorrange=clims, kwargs...)
    Colorbar(fig[1, 2];colorrange=clims, kwargs...)
    fig
end
end