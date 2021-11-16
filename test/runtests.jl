include("C:\\Users\\r2d2go\\dumgit\\Tmatrix.jl\\src\\tmatrix_complex.jl")
import Zygote

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 1
k1_r = 1e7; k1_i = 0.0
k2_r = 1.5e7; k2_i = 1e3
k1 = Complex(k1_r, k1_i)
k2 = Complex(k2_r, k2_i)
calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2; rotationally_symmetric=true, symmetric_about_plane_perpendicular_z=false)

function Tmatrix_spheroid_simple(rx, rz)
    return calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2; rotationally_symmetric=true, symmetric_about_plane_perpendicular_z=false);
end
