module NeumannKelvinWGLMakieExt

if isdefined(Base, :get_extension)
    using WGLMakie; WGLMakie.activate!(inline=false)
else
    using ..WGLMakie; WGLMakie.activate!(inline=false)
end

using NeumannKelvin

# Generate full triangle mesh and color array
function viz(panels, values; vectors=panels.n, colormap=:viridis)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=:data)

    # Normalize and map to color
    vmin, vmax = extrema(values)
    norm_vals = (values .- vmin) ./ (vmax - vmin + eps())
    colors = cgrad(colormap)[norm_vals]

    # Mesh geometry (triangle list) and colors
    triangles = mapreduce(vcat, panels) do panel
        p = panel.⛶
        [p[1], p[2], p[3], p[4], p[3], p[2]]
    end
    tri_colors = mapreduce(c -> fill(c, 6), vcat, colors)
    mesh!(ax, triangles; color=tri_colors)

    # Normals & color bar
    arrows!(ax, components(panels.x)..., components(vectors)...; color=colors)
    Colorbar(fig[1, 2], limits=(vmin, vmax); colormap)
    fig
end
export viz

end