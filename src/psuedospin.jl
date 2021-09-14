using GLMakie
using FileIO
using LinearAlgebra


include("utils.jl")
include("tmatrix_complex.jl")
#input format: scattering cross section magnitude (R), emiss cross section ((x,y) point_list)
function main()
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

    function calculate_Tmatrix_for_spheroid(
        rx::R,
        rz::R,
        n_max::Int,
        k1::Complex,
        k2::Complex;
        n_θ_points = 10,
        n_ϕ_points = 20,
        HDF5_filename = nothing,
        rotationally_symmetric = false,
        symmetric_about_plane_perpendicular_z = false,
        BigFloat_precision = nothing,
    ) where {R <: Real}
    
        # create a grid of θ_ϕ
        θ_array, ϕ_array = meshgrid_θ_ϕ(
            n_θ_points,
            n_ϕ_points;
            min_θ = 1e-16,
            min_ϕ = 1e-16,
            rotationally_symmetric = rotationally_symmetric,
        )
    
        # calculate r and n̂ for the geometry
        r_array, n̂_array = ellipsoid(rx, rz, θ_array)
        
        println("N ARRAY NOW")
        println(size(n̂_array))
        println(n̂_array)
        println("R ARRAY NOW")
        println(size(r_array))
        println(r_array)
        # calculate T-matrix
        k1r_array = k1 .* r_array
        k2r_array = k2 .* r_array
        T = T_matrix(
            n_max,
            k1,
            k2,
            k1r_array,
            k2r_array,
            r_array,
            θ_array,
            ϕ_array,
            n̂_array;
            HDF5_filename = HDF5_filename,
            rotationally_symmetric = rotationally_symmetric,
            symmetric_about_plane_perpendicular_z = symmetric_about_plane_perpendicular_z,
            BigFloat_precision = BigFloat_precision,
        )
        return T
    end

    function calculate_Tmatrix_for_any(
        point_list,
        face_list, #make this better typed
        n_max::Int,
        k1::Complex{R},
        k2::Complex{R};
        HDF5_filename = nothing,
        rotationally_symmetric = false,
        symmetric_about_plane_perpendicular_z = false,
        BigFloat_precision = nothing,
    ) where {R <: Real}
        
        spherical_list = convert_to_spherical(point_list) #format is [r, θ, ϕ], use coordinatetransformations.jl to convert
    
        #TODO: bundle these into a new struct and make that not break
        sls = size(spherical_list,1)
        r_array = reshape([spherical_list[row, 1] for row in 1:sls],(convert(Int,(2*sls)^.5), convert(Int,(2*sls)^.5/2)))
        θ_array = reshape([spherical_list[row, 2] for row in 1:sls],(convert(Int,(2*sls)^.5), convert(Int,(2*sls)^.5/2)))
        ϕ_array = reshape([spherical_list[row, 3] for row in 1:sls],(convert(Int,(2*sls)^.5), convert(Int,(2*sls)^.5/2)))
        
        # calculate r and n̂ for the geometry
    
        n̂_array = reshape([zeros(3) for i in 1:size(point_list, 1)],(convert(Int,(2*sls)^.5), convert(Int,(2*sls)^.5/2)))#copying so that size is same

    
        for row in 1:size(face_list,1)
            face = face_list[row, :]
            for i in 1:size(face,1)
                vertex1 = face[i]
                vertex2 = face[i%size(face,1)+1]
                vertex3 = face[(i+1)%size(face,1)+1]
                edge1 = spherical_list[vertex1,:]-spherical_list[vertex2,:]
                edge2 = spherical_list[vertex1,:]-spherical_list[vertex3,:]
                n̂_array[vertex1] += cross(edge1, edge2)
            end
        end

        normalize(n̂_array)
        # calculate T-matrix
        k1r_array = k1 .* r_array
        k2r_array = k2 .* r_array
        T = T_matrix(
            n_max,
            k1,
            k2,
            k1r_array,
            k2r_array,
            r_array,
            θ_array,
            ϕ_array,
            n̂_array;
            HDF5_filename = HDF5_filename,
            rotationally_symmetric = rotationally_symmetric,
            symmetric_about_plane_perpendicular_z = symmetric_about_plane_perpendicular_z,
            BigFloat_precision = BigFloat_precision,
        )
        return T
    end
    
    #=
    function calculate_T(
        face_list, #make this better typed
        point_list
    ) 
            
        spherical_list = convert_to_spherical(point_list) #format is [r θ ϕ], use coordinatetransformations.jl to convert

        #TODO: bundle these into a new struct and make that not break
        r_array = [spherical_list[row, 1] for row in 1:size(spherical_list,1)]
        θ_array = [spherical_list[row, 2] for row in 1:size(spherical_list,1)]
        ϕ_array = [spherical_list[row, 3] for row in 1:size(spherical_list,1)]

        # calculate r and n̂ for the geometry

        n̂_array = zeros(size(point_list)) #copying so that size is same

        

        for row in 1:size(face_list,1)
            face = face_list[row, :]
            for i in 1:size(face,1)
                vertex1 = face[i]
                vertex2 = face[i%size(face,1)+1]
                vertex3 = face[(i+1)%size(face,1)+1]
                edge1 = spherical_list[vertex1,:]-spherical_list[vertex2,:]
                edge2 = spherical_list[vertex1,:]-spherical_list[vertex3,:]
                n̂_array[vertex1, :] += cross(edge1, edge2)
            end
        end
        #TODO #URGENT #IMPORTANT: Make this use proper Tmatrix instead of whatever this is
        T = randn(1, size(point_list, 1), 2)
        for i in 1:size(point_list,1)
            #element = normalize(element) #TODO make sure this works, use normalize!(element) later if at all
            element = norm(n̂_array[i, :])
            T[1, i, 2] = element
            T[1, i, 1] = i
        end
        
        return T
    end
    =#


    function plot_mesh_and_emiss(i, point_list, granularity)

        mesh_point_list, mesh_face_list = make_mesh(granularity, point_list)
        rx = 3
        rz = 2
        n_max = 2
        k1_r = 1e7
        k1_i = 0.0
        k2_r = 1.5e7
        k2_i = 1e3

        k1 = Complex(k1_r, k1_i)
        k2 = Complex(k2_r, k2_i)
        T = calculate_Tmatrix_for_any(mesh_point_list, mesh_face_list, n_max, k1, k2)

        scattering = get_OrentationAv_scattering_CrossSections_from_Tmatrix(T)
        
        data = [i, scattering]

        f = Figure(
        resolution = (1000, 700))
        ga = f[1, 1] = GridLayout()
        gc = f[1, 2] = GridLayout()
        axtop = Axis(ga[1, 1])
        labels = ["scattering cross section"]
        
        for (label, col) in zip(labels, eachslice(data, dims = 1))
            scatter!(axtop, col, label = label)
        end

        save_as_obj(mesh_point_list, mesh_face_list, i)

        brain = load(assetpath("C:/Users/r2d2go/dum/Tmatrix.jl/data/mesh_$(i).obj"))
        Axis3(gc[1, 1], title = "Mesh")
        m = mesh!(
            brain,
            color = :blue,
        )

        display(f)
        
    end
    list_of_point_lists = [[0 0; 0 10; 4 10-i/2; 4 i/2] for i in 1:10]
    for i in 1:10
        
        granularity = 32
        framerate = 5
        point_list = list_of_point_lists[i]
        plot_mesh_and_emiss(i, point_list, granularity)
        sleep(1/framerate)
    end

end

main()
