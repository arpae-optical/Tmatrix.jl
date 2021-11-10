using Tmatrix
using Zygote
using Plots
using Distributions

function objective_function(r_array_input, θ_array_input, wavelens, target_emiss)
    r_θ_array = Tmatrix.quadruple_mesh_density(r_array_input, θ_array_input)
    r_array = r_θ_array[:, 1]
    θ_array = r_θ_array[:, 2]
    ϕ_array = zeros(size(θ_array))

    T = [
        Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
            n_max,
            w,
            input_unit,
            Eps_r_r_1,
            Eps_r_i_1,
            Mu_r_r_1,
            Mu_r_i_1,
            Eps_r_r_2,
            Eps_r_i_2,
            Mu_r_r_2,
            Mu_r_i_2,
            (collect(r_array)),
            θ_array,
            ϕ_array,
            rotationally_symmetric,
            symmetric_about_plane_perpendicular_z,
            BigFloat_precision,
        ) for w in wavelens
    ]
    k1_complex = [
        Tmatrix.get_WaveVector(
            w;
            input_unit = input_unit,
            Eps_r = Complex(Eps_r_r_1, Eps_r_i_1),
            Mu_r = Complex(Mu_r_r_1, Mu_r_i_1),
        ) for w in wavelens
    ]

    surface_area = Tmatrix.calculate_surface_area_of_axisymmetric_particle(r_array, θ_array)

    emiss = [
        Tmatrix.get_OrentationAv_emissivity_from_Tmatrix(t, k1, surface_area) for
        (t, k1) in zip(T, k1_complex)
    ]
    loss = sum(abs.(emiss - target_emiss) .^ 2) / length(emiss)
    return loss
end

function ∂objective_function(r_array, θ_array, target_wavelens, target_emiss)
    return Zygote.gradient(
        (r_array, θ_array) ->
            objective_function(r_array, θ_array, target_wavelens, target_emiss),
        (collect(r_array)),
        (collect(θ_array)),
    )
end

function rand_target_emiss(n::Integer = 100)
    wavelens = collect(LinRange(1e-6, 20e-6, n))
    emiss = rand(Uniform(0.8, 1.0), n)
    return (wavelens, emiss)
end

# Particle size should be comparable to wavelen.
const input_unit = "m"
# TODO issue of matrix inverse for n≥4, see emails
const n_max = 3
const Eps_r_r_1 = 1.0
const Eps_r_i_1 = 0.0
const Eps_r_r_2 = 1.5
const Eps_r_i_2 = 0.01
const Mu_r_r_1 = 1.0
const Mu_r_i_1 = 0.0
const Mu_r_r_2 = 1.0
const Mu_r_i_2 = 0.0
const rotationally_symmetric = true
const symmetric_about_plane_perpendicular_z = false
const BigFloat_precision = nothing
# Set small to avoid radii growing too big, past wavelength, and T matrix
# blowing up in Bessel function calculation.
const learning_rate = 0.5e-5
const num_pts = 20
const num_ctrl_pts = 20
const target_wavelens, target_emiss = rand_target_emiss(num_pts)

θ_array = collect(LinRange(1e-6, pi - 1e-6, num_ctrl_pts))
angular_jitter = [0.0, rand(Uniform(-1e-5, 1e-5), num_ctrl_pts - 2)..., 0.0]
θ_array += angular_jitter

# 1e-5 is to pick a reasonable starting size
r_array = rand(Normal(1e-5, 1e-5 / 2), size(θ_array))

loss_array = []

for n_iteration in 1:50
    loss_here = objective_function(r_array, θ_array, target_wavelens, target_emiss)
    ∂loss_r, ∂loss_θ = ∂objective_function(r_array, θ_array, target_wavelens, target_emiss)
    # ∂loss_r = clamp.(∂loss_r, -1e-5, 1e-5)
    # TODO use seq of offsets to avoid negatives
    # ∂loss_θ = clamp.(∂loss_θ, -.5, 0.5)

    # zero out the first and last control points angle
    ∂loss_θ[1] = 0.0
    ∂loss_θ[length(∂loss_θ)] = 0.0
    global r_array = r_array .- learning_rate .* ∂loss_r
    global θ_array = θ_array .- learning_rate .* ∂loss_θ
    append!(loss_array, loss_here)

    println()
    println("iteration = $n_iteration")
    println("loss_here = $loss_here")
    println("∂loss_r = $∂loss_r")
    println("∂loss_θ = $∂loss_θ")
    println("r = $r_array")
    println("θ = $θ_array")

    xyz = vcat(
        Tmatrix.convert_coordinates_Sph2Cart.(r_array, θ_array, zeros(size(r_array)))...,
    )
    p1 = plot!(
        xyz[:, 1],
        xyz[:, 3],
        aspect_ratio = :equal,
        label = "particle after $n_iteration iterations",
        title = "scattering cross section = $loss_here m",
    )
    p2 = plot(
        1:length(loss_array),
        loss_array,
        xlabel = "iteration #",
        ylabel = "scattering cross section (m)",
    )
    fig = plot(p1, p2, layout = (1, 2), size = (1200, 800))
    mkpath("cache/iteration_particle_plots/maximizing_emissivity")
    # savefig(
    # fig,
    # "cache/iteration_particle_plots/maximizing_emissivity/particle_geom_iteration_$(n_iteration).png",
    # )
end
