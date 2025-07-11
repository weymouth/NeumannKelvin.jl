module NeumannKelvinPlotsExt
using NeumannKelvin,Plots
function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=0.3panels.n, kwargs...)
    @warn "Using fallback Plots-based panel plotting. Load GLMakie or WGLMakie instead for more functionality." maxlog=1
    (xmn,xmx),(ymn,ymx),(zmn,zmx)=extrema.(components(panels.x))
    xc,yc,zc = 0.5xmx+0.5xmn,0.5ymx+0.5ymn,0.5zmx+0.5zmn
    h=1.05max(xmx-xc,ymx-yc,zmx-zc)
    scatter(components(panels.x),marker_z=values;label="",kwargs...)
    !isnothing(vectors) && quiver!(components(panels.x),quiver=components(vectors),
            color=:grey,xlims=(xc-h,xc+h),ylims=(yc-h,yc+h),zlims=(zc-h,zc+h))
    plot!()
end
@recipe function f(table::Table)
    cols = columns(table)
    col_names = columnnames(table)
    
    if length(col_names) < 2
        error("Table must have at least 2 columns (x-axis and at least one y-series)")
    end
    
    # First column as x-axis
    x_data = cols[1]
    x_name = string(col_names[1])
    
    # Set x-axis label
    xlabel --> x_name
    
    # Plot each remaining column as a separate series
    for i in 2:length(col_names)
        y_data = cols[i]
        y_name = string(col_names[i])
        
        @series begin
            label --> y_name
            x_data, y_data
        end
    end
end
end
