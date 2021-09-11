using GLMakie
using FileIO
#input format: scattering cross section magnitude (R), emiss cross section ((x,y) point_list)
function main()
    granularity = 30
    framerate = 5
    
    
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
        face_list = []
        for u in 0:num_points-1
            for v in 0:granularity-1
                p1 = granularity*u+v+1
                p2 = mod(granularity*(u + 1) + v, num_points*granularity)+1
                p3 = mod(granularity*(u + 1) + mod(v + 1, granularity), num_points*granularity)+1
                p4 = mod(granularity*u + mod(v + 1, granularity), num_points*granularity)+1
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

    function save_as_obj(point_list, face_list, mesh_index)
        open("data/mesh_$(mesh_index).obj", "w") do mesh_target
            for row in 1:size(point_list,1)
                write(mesh_target, "v $(point_list[row, 1]) $(point_list[row, 2]) $(point_list[row, 3]) \n")
            end
            for row in 1:size(face_list,1)
                write(mesh_target, "f $(face_list[row, 1]) $(face_list[row, 2]) $(face_list[row, 3]) \n")
            end
        end
    end

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



    function plot_mesh_and_emiss(i)

        point_list = [0 0; 0 10; 4 10-i/2; 4 i/2]

        f = Figure(
        resolution = (1000, 700))
        ga = f[1, 1] = GridLayout()
        gb = f[2, 1] = GridLayout()
        gcd = f[1:2, 2] = GridLayout()
        gc = gcd[1, 1] = GridLayout()
        gd = gcd[2, 1] = GridLayout()

        axtop = Axis(ga[1, 1])
        axmain = Axis(ga[2, 1], xlabel = "before", ylabel = "after")
        axright = Axis(ga[2, 2])

        labels = ["treatment", "placebo", "control"]
        data = randn(3, 100, 2) .+ [1, 3, 5]

        for (label, col) in zip(labels, eachslice(data, dims = 1))
            scatter!(axmain, col, label = label)
            density!(axtop, col[:, 1])
            density!(axright, col[:, 2], direction = :y)
        end

        Axis3(gc[1, 1], title = "Brain activation")


        mesh_point_list, mesh_face_list = make_mesh(granularity, point_list)


        save_as_obj(mesh_point_list, mesh_face_list, i)

        brain = load(assetpath("C:/Users/r2d2go/dum/Tmatrix.jl/data/mesh_$(i).obj"))
        Axis3(gc[1, 1], title = "Brain activation")
        m = mesh!(
            brain,
            color = :blue,
        )

        display(f)
        
    end
    for i in 1:10
        plot_mesh_and_emiss(i)
        sleep(1/framerate)
    end
end

main()
