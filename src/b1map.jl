using Zygote
using ADTypes: AutoReverseDiff, AutoZygote
using Optim: optimize, LBFGS 
import Optim

export roughness_penalty, regularizer, log_loss, unpack, b1_fit, complex_gaussian_noise, DAM

"""
Compute the roughness penalty for a 2d image with vectorized operations. See eqation 41 in regularized B1 mapping paper.

# In
- `im` 2d image of size (N1, N2)
- `nl_mls` Coordinate offsets to compute roughness. Defaults to top, bottom, left, and right.

# Out
- `penalty` The roughness penalty for the image.
"""
function roughness_penalty(
        im::AbstractMatrix;
        nl_mls::Tuple = ((1, 0), (0, 1))
    )
    N1, N2 = size(im)
    penalty = zero(real(eltype(im)))
    
    for offset in nl_mls
        n_l, m_l = offset[1], offset[2]
        abs_n, abs_m = abs(n_l), abs(m_l)

        # Slice views to compute 2*z[n,m] - z[n-n_l, m-m_l] - z[n+n_l, m+m_l] without loops
        center = @view im[1+abs_n:N1-abs_n, 1+abs_m:N2-abs_m]
        left   = @view im[1:N1-2*abs_n,     1:N2-2*abs_m]
        right  = @view im[1+2*abs_n:N1,     1+2*abs_m:N2]

        penalty += sum(abs2, 2 .* center .- left .- right)
    end

    return penalty
end

"""
Compute the total roughness penalty over each coil slice using sum generator syntax. See equation 6.

# In
- `zks` Unknown b1 feild map of size (N1, N2, K) where N1 and N2 are spatial dimensions and K is the coil number.

# Out
- `cost` Returns the total roughness penalty across each coil.
"""
function regularizer(zks::AbstractArray)
    K = size(zks, 3)
    return sum(roughness_penalty(@view zks[:, :, k]) for k in 1:K)
end

"""
Helper function. See equation 14.
"""
function l_equation_elem(y, f, a, b, F::Function)
    g_val = F(complex(a, b))
    return 0.5 * abs2(y - f * g_val)
end

"""
Vectorized loss function. See equation 23.

# In
- `zks` Unknown b1 feild map of size (N1, N2, K) where N1 and N2 are spatial dimensions and K is the coil number.
- `fjs` Underlying object transverse magnetization with shape (N1, N2)
- `Chi` Matrix of scaling factors of size (M, K) where m is the measurement number and k is the coil number.
- `Y` Reconstructed images of shape (N1, N2, M) where M is the measurement number
- `F`

# Out
- `loss` the loss for the given zks.
"""
function log_loss(
        zks::AbstractArray, 
        fjs::AbstractArray, 
        Chi::AbstractArray,
        Y::AbstractArray,
        F::Function,
    )
    N1, N2, K = size(zks)
    M = size(Chi, 1)

    # Reshape and project via Chi
    Chi_z = reshape(reshape(zks, N1 * N2, K) * Chi', N1, N2, M)

    # Fused calculation over broadcasted arrays
    l_array = l_equation_elem.(Y, fjs, real.(Chi_z), imag.(Chi_z), F)
    return real(sum(l_array))
end

"""
Objective function for optimization combining log loss and regularization.

    x0 = [real.(z_hat_flat); imag.(z_hat_flat); real.(f_hat_flat); imag.(f_hat_flat)]
# In
- `params` A vector of parameters to optimize over. We expect this to be a vector where the first 2NKM elements are the real and imaginary parts of z_hat and the next 2N elements are the real and imaginary parts of f.
- `zdims` (N1, N2, K)
- `fdims` (N1, N2)
- `Beta` Regularization coefficient between 0 and 1
- `Y` Reconstructed images of shape (N1, N2, M) where M is the measurement number
- `Chi` Matrix of scaling factors of size (M, K) where m is the measurement number and k is the coil number.
- `F`

# Out
- `cost` a real valued cost including both the regularized roughness penalty and log loss
"""
function psi(
        params::AbstractVector, 
        zdims::Tuple, 
        fdims::Tuple, 
        Beta::Real, 
        Y::AbstractArray,
        Chi::AbstractArray,
        F::Function, 
    )
    zks, fjs = unpack(params, zdims, fdims)
    return log_loss(zks, fjs, Chi, Y, F) + Beta * regularizer(zks)
end

"""
Unpack optimized parameter vector into complex 3D and 2D arrays.

# In
- `params` (2NK + 2N) vector where the first 2NK elements are 
the real and imaginary components of the b1 feild and the last 2N 
are the real and imaginary components of the underlying transervse magnetization.
- `zdims` (N1, N2, K)
- `fdims` (N1, N2)
"""
function unpack(
        params::AbstractVector,
        zdims::Tuple,
        fdims::Tuple 
    )
    N1, N2, K = zdims
    N1 == fdims[1] || throw(ArgumentError("N's don't match."))
    N2 == fdims[2] || throw(ArgumentError("D's don't match."))

    fn = N1 * N2
    zn = N1 * N2 * K

    # Direct vector slicing creates clean, Zygote-friendly adjoints
    zks_real = params[1:zn]
    zks_im   = params[zn+1 : 2*zn]
    zks      = reshape(complex.(zks_real, zks_im), zdims)

    fjs_real = params[2*zn+1 : 2*zn+fn]
    fjs_im   = params[2*zn+fn+1 : 2*zn+2*fn]
    fjs      = reshape(complex.(fjs_real, fjs_im), fdims)

    return zks, fjs
end

"""
Helper function. See appendix C.
"""
function H(z::AbstractArray, F::Function)
    return F.(z) ./ (exp.(im .* z) .+ 1e-12)
end

"""
Compute the initial estimate of x from the reconstructed images.

# In
- `Y` Reconstructed images of shape (N1, N2, M) where M is the measurement number
- `K` The number of coils
- `F`

# Out
- The optimal starting point for the optimziation algorithm
"""
function compute_x_hat(
        Y::AbstractArray,
        K::Int,
        F::Function
    )
    T = eltype(Y)
    N1, N2, M = size(Y)

    x_hat_mags = zeros(T, N1, N2, K)
    for m in 1:K
        ratio = real(0.5 .* abs.(Y[:, :, m + K] ./ (Y[:, :, m] .+ 1e-12)))
        x_hat_mags[:, :, m] = acos.(clamp.(ratio, zero(real(T)), one(real(T)) - 1e-7))
    end

    x_hat_angles = zeros(eltype(Y), N1, N2, K)
    for m in 1:K
        x_hat_angles[:, :, m] = angle.(Y[:, :, m]) - angle.(H(abs.(x_hat_mags[:, :, K]), F))
    end

    return x_hat_mags .* exp.(im .* x_hat_angles)
end

"""

# In
- `z_hat` The inital guess for the b1 feild of size (N1, N2, K) where k is the coil number.
- `Chi` Matrix of scaling factors of size (M, K) where m is the measurement number and k is the coil number.

# Out
- Estimated composite map of size (N1, N2, M)
"""
function compute_composite_map(
        z_hat::AbstractArray,
        chi::AbstractArray
    )
    N1, N2, K = size(z_hat)
    M = size(chi, 1)
    K == size(chi, 2) || throw(ArgumentError("K's don't match."))

    x = zeros(eltype(z_hat), N1, N2, M)
    for k in 1:K
        z_flat = reshape(z_hat[:, :, k], :, 1)  # (N1*N2) x 1
        x_perm = chi[:, k] * z_flat'            # M x (N1*N2)
        x_perm = reshape(x_perm, M, N1, N2)       # M x N1 x N2
        x += permutedims(x_perm, (2, 3, 1))      # N1 x N2 x M
    end

    return x
end

"""
Calculate the starting f for the optimization algorithm.

    # In
    - `Y` Reconstructed images of shape (N1, N2, M) where M is the measurement number
    - `z_hat` The inital guess for the b1 feild of size (N1, N2, K) where k is the coil number.
    - `Chi` Matrix of scaling factors of size (M, K) where m is the measurement number and k is the coil number.

    # Out
    - The inital guess for the underlying transverse magnetization of size (N1, N2)
"""
function compute_f_hat(
        Y::AbstractArray,
        z_hat::AbstractArray,
        chi::AbstractArray,
        F::Function
    )
    N1, N2, M = size(Y)
    xjm = compute_composite_map(z_hat, chi)

    top = zeros(eltype(Y), N1, N2)
    bottom = zeros(eltype(Y), N1, N2)

    for m in 1:M
        top += real.(conj.(Y[:, :, m]) .* F.(xjm[:, :, m]))
        bottom += abs.(F.(xjm[:, :, m])) .^ 2
    end

    return top ./ bottom
end

"""
Perform the regularized b1 fit.

# In
- `Beta` Regularization coefficient
- `Y` Reconstructed images of shape (N1, N2, M) where M is the measurement number
- `Chi` Matrix of scaling factors of size (M, K) where m is the measurement number and k is the coil number.
- `F`
- `MAX_ITTER`

# Out
- `zk_opt` The estimated b1 feild of size (N1, N2, K)
- `fj_opt` The estimated transverse magnetization of size(N1, N2)
"""
function b1_fit(
        Beta::Real,
        Y::AbstractArray,
        Chi::AbstractArray,
        F::Function,
        MAX_ITTER::Int
    )
    N1, N2, M = size(Y)
    K = size(Chi, 2) 
    zdims = (N1, N2, K)
    fdims = (N1, N2)

    # Error checking
    M == size(Chi, 1) || throw(ArgumentError("M's don't match."))
    M == 2 * K || throw(ArgumentError("Chi must be M x M/2."))
    
    half_M = div(M, 2)
    Chi_tilda = Chi[1:half_M, :]
    isapprox(Chi[half_M + 1:end, :], 2 * Chi_tilda) || throw(ArgumentError("The bottom half of Chi must be twice the top half of Chi."))
    
    # Initial guess estimation
    x_hat = compute_x_hat(Y, K, F)
    x_hat_perm = PermutedDimsArray(x_hat, (3, 1, 2))
    x_hat_flat = reshape(x_hat_perm, K, :)
    z_hat_flat = Chi_tilda \ x_hat_flat
    z_hat_perm = reshape(z_hat_flat, K, N1, N2)

    z_hat = permutedims(z_hat_perm, (2, 3, 1))
    f_hat = compute_f_hat(Y, z_hat, Chi, F)

    z_hat_flat = reshape(z_hat, :)
    f_hat_flat = reshape(f_hat, :)

    x0 = [real.(z_hat_flat); imag.(z_hat_flat); real.(f_hat_flat); imag.(f_hat_flat)]

    cost(x::AbstractVector) = psi(x, zdims, fdims, Beta, Y, Chi, F)
    options = Optim.Options(
        iterations = MAX_ITTER,
        show_trace = true
    )
    
    out = Optim.optimize(cost, x0, LBFGS(), options; autodiff = AutoZygote())
    zk_opt, fj_opt = unpack(out.minimizer, zdims, fdims)

    return zk_opt, fj_opt
end