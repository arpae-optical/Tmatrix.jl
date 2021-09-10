using GLMakie
#input format: scattering cross section magnitude (R), emiss cross section ((x,y) point_list)
function main()
    granularity = 10
    framerate = 2
    
    #=
    function convert_to_spherical(point_list)
        spherical_list = []
        for row in 1:size(point_list,1)
            point_x = point_list[row, 1]
            point_y = point_list[row, 2]
            point_z = point_list[row, 3]
            r = point_x^2 + point_y^2 + point_z^2

            if point_y == 0
                θ = 0
            else
                θ = atan(point_x/point_y)
            end

            if point_y == 0        
                ϕ = 0
            else   
                ϕ = acos(point_z/r)
            end

            if row == 1
                spherical_list = [r θ ϕ]
            else
                spherical_list = vcat(spherical_list, [r θ ϕ])
            end
        end
        return(spherical_list)
    end
=#

    function draw_mesh(points, faces)
        scene = mesh(points, faces, shading = false)
    end

    function make_mesh(granularity::Int, point_list)
        num_points = size(point_list,1)
        new_point_list = []
        for row in 1:num_points
            for arc in 0:granularity-1
                theta = arc*2*pi/granularity
                point_x = cos(theta)*point_list[row,1]
                point_y = sin(theta)*point_list[row,1]
                point_z = point_list[row,2]
                if new_point_list == []
                    new_point_list = [point_x point_y point_z]
                else
                    new_point_list = vcat(new_point_list, [point_x point_y point_z])
                end
            end
        end
        print(size(point_list))
        face_list = []
        for v in 4:4
            for u in 0:num_points-1
                p1 = granularity*u+v+1
                p2 = mod(granularity*(u + 1) + v, num_points*granularity)+1
                p3 = mod(granularity*(u + 1) + mod(v + 1, num_points), num_points*granularity)+1
                p4 = mod(granularity*u + mod(v + 1, num_points), num_points*granularity)+1
                #=
                p1 = num_points*u+v+1
                p2 = mod(num_points*(u + 1) + v, num_points*granularity)+1
                p3 = mod(num_points*(u + 1) + mod(v + 1, granularity), num_points*granularity)+1
                p4 = mod(num_points*u + mod(v + 1, granularity), num_points*granularity)+1

                =#
                if face_list == []
                    face_list =
                        [
                            p1 p2 p3;
                            p1 p3 p4
                        ]
                    
                else
                    face_list = vcat(
                        face_list,
                        [
                            p1 p2 p3;
                            p1 p3 p4
                        ]
                    )
                end
            end
        end

        return (new_point_list, face_list)
    end

#=
    function calculate_T(
        face_list::Vector{Any}, #make this better typed
        point_list::Vector{Vector{Float64}}
    ) 
            
        spherical_list = convert_to_spherical(point_list) #format is [r, θ, ϕ], use coordinatetransformations.jl to convert

        #TODO: bundle these into a new struct and make that not break
        r_array, θ_array, ϕ_array = ([element[1] for element in spherical_list], [element[2] for element in spherical_list], [element[3] for element in spherical_list])

        # calculate r and n̂ for the geometry

        n̂_array = point_list #copying so that size is same

        for element in n̂_array
            element = [0,0,0]
        end

        for face in face_list
            for (vertex, i) in enumerate(face)
                n_array[vertex] += vertex × point_list[face[(i+1)%size(face)]] #TODO this should be edges not vertices
            end
        end

        for element in n̂_array
            element = normalize(element) #TODO make sure this works, use normalize!(element) later if at all
        end

        T = sum(n̂_array)
        return T
    end
=#


    for i in 1:1

        point_list = [1 1 1; 1 8 1; 3 3 3; 6 2 5]

        mesh_point_list, mesh_face_list = make_mesh(granularity, point_list)
        print(mesh_point_list, mesh_face_list)
        scene = mesh(mesh_point_list, 
        mesh_face_list, 
        color = :blue, 
        figure = (resolution = (1000, 1000),),
        figure_padding = 1)
        #scene, layout = layoutscene()
        #ax = layout[1, 1] = Axis3(scene)
        #limits!(ax, -8, 8, -8, 8, -8, 8)
        display(scene)
        sleep(1/framerate)
        
    end
end

main()
