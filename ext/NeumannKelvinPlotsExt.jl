module NeumannKelvinPlotsExt
using NeumannKelvin,Plots
function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=0.3panels.n, kwargs...)
    @warn "Using fallback Plots-based panel plotting. Load GLMakie or WGLMakie instead for more functionality." maxlog=1
    scatter(components(panels.x),marker_z=values;label="",kwargs...)
    quiver!(components(panels.x),quiver=components(vectors),color=:grey)
end
end
