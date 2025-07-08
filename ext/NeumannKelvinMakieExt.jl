module NeumannKelvinMakieExt
using NeumannKelvin,Makie
using Makie.GeometryBasics
# Generate full triangle mesh and color array
function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=0.3panels.n, clims = extrema(values), kwargs...)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=:data)

    # Mesh geometry (vertices, normals, and colors)
    face = decompose(QuadFace{GLIndex},Tessellation(Rect(0,0,1,1),(2,2))) # 2x2 quad type
    unwrap(a) = Point3f.(map(i->a[i],SA[1,2,4,3])) # flattens our 2x2 data counter-clockwise
    quad(panel) = GeometryBasics.Mesh(unwrap(panel.xᵤᵥ), face, normal=unwrap(panel.nᵤᵥ))
    mesh!(ax, quad.(panels); color=values, colorrange=clims, kwargs...)

    # Normals & color bar
    !isnothing(vectors) && arrows3d!(ax, components(panels.x)..., components(vectors)...; 
        lengthscale=0.5, color=values, colorrange=clims, kwargs...)
    Colorbar(fig[1, 2];colorrange=clims, kwargs...)
    fig
end
end