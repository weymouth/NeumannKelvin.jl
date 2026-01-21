module NeumannKelvinPlotsExt
using NeumannKelvin,Plots
import NeumannKelvin: viz,cₚu,xyζ,QuadKernel

# Free surface viz function
function viz(sys::Union{FSPanelSystem,NKPanelSystem},::Val{ζ};kwargs...)
    x,y,z = xyζ(sys)
    ζmax  = maximum(abs, z)
    surface(x,y,z';colormap = :balance, clims = (-ζmax,ζmax), 
        colorbar=true, colorbar_title="ζ")
end

# Generate mesh from Table data
function viz(panels::Union{Table,PanelTree}, values=panels.dA; vectors=panels.n,
                           vscale=1, label="", clims=extrema(values), colormap=:viridis, kwargs...)
    # Get colors from colormap
    cmap = cgrad(colormap)
    color = [cmap[(v - clims[1]) / (clims[2] - clims[1])] for v in values]
    tricolor = eltype(panels.kernel)==QuadKernel ? mapreduce(c->[c;c],vcat,color) : color

    # Collect all vertices
    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    connections_i = Int[]
    connections_j = Int[]
    connections_k = Int[]

    for panel in panels
        addtri!(xs,ys,zs,connections_i,connections_j,connections_k,panel.verts,panel.kernel)
    end

    # Invisible plot for colorbar
    cx,cy,cz = components(panels.x)
    scatter3d(cx,cy,cz;markersize=0,fill_z=values,clims,c=colormap,
            colorbar=true,colorbar_title=label,legend=false)

    # Main plot
    mesh3d!(xs, ys, zs; color = tricolor, linewidth=0,
            connections=(connections_i, connections_j, connections_k), kwargs...)

    # Add normal vectors if provided
    if !isnothing(vectors)
        # Extract panel centers and normal vectors
        nx,ny,nz = components(0.05vscale .* vectors)
        quiver!(cx, cy, cz; color=:grey, quiver=(nx, ny, nz), arrow=:closed, linewidth=1.5)
    end

    # set plot limits, labels
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = extrema.((xs,ys,zs))
    cxm,cym,czm = (xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2
    Δ = maximum((xmax-xmin, ymax-ymin, zmax-zmin))
    plot!(xlabel="x",ylabel="y",zlabel="z",
        xlims=(cxm-Δ/2, cxm+Δ/2),ylims=(cym-Δ/2, cym+Δ/2),zlims=(czm-Δ/2, czm+Δ/2))
end
addtri!(x,y,z,i,j,k,verts,::QuadKernel) = (
    addtri!(x,y,z,i,j,k,verts[1:3]); addtri!(x,y,z,i,j,k,verts[[1,3,4]]))
function addtri!(x,y,z,i,j,k,verts,args...)
    for v in verts
        push!(x, v[1]); push!(y, v[2]); push!(z, v[3])
    end

    # Connections for this triangle (0-indexed)
    base = length(x)-3
    push!(i, base); push!(j, base + 1); push!(k, base + 2)
end
end