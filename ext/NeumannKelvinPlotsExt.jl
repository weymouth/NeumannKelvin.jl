module NeumannKelvinPlotsExt
using NeumannKelvin,Plots
function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=0.3panels.n, kwargs...)
    @warn "Using fallback Plots-based panel plotting. Load GLMakie or WGLMakie instead for more functionality." maxlog=1
    (xmn,xmx),(ymn,ymx),(zmn,zmx)=extrema.(components(panels.x))
    xc,yc,zc = 0.5xmx+0.5xmn,0.5ymx+0.5ymn,0.5zmx+0.5zmn
    h=1.05max(xmx-xc,ymx-yc,zmx-zc)
    scatter(components(panels.x),marker_z=values;label="",kwargs...)
    quiver!(components(panels.x),quiver=components(vectors),color=:grey,
            xlims=(xc-h,xc+h),ylims=(yc-h,yc+h),zlims=(zc-h,zc+h))
end
end
