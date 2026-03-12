using LinearAlgebra



# --- Cubic Spline Implementation ---

# Define the Spline object to store spline information
struct CubicSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    a::Vector{Float64} # a[i] = y[i]
    b::Vector{Float64} # Coefficients for (x-x[i])
    c::Vector{Float64} # Coefficients for (x-x[i])^2
    d::Vector{Float64} # Coefficients for (x-x[i])^3
    bc_type::String
    extrapolate::Bool
end

"""
    CubicSpline(x, y; bc_type="not-a-knot", extrapolate=true)

Constructs a CubicSplineInterpolation object from given x and y data.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `bc_type`: Boundary condition type. Can be "natural", "clamped", or "not-a-knot".
               Defaults to "natural". "clamped" is not yet fully implemented and will behave like natural.
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true.

# Returns
- A CubicSplineInterpolation object.
"""
function CubicSpline(x::Vector{Float64}, y::Vector{Float64}; bc_type::String="not-a-knot", extrapolate::Bool=true)
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n <= 2
        throw(ArgumentError("At least 3 data points are required for cubic spline interpolation"))
    end
    if !issorted(x)
        throw(ArgumentError("x must be strictly increasing"))
    end
    bc_type_lower = lowercase(bc_type)
    if bc_type_lower ∉ ["natural", "clamped", "not-a-knot"]
        @warn "Boundary condition type '$bc_type' not recognized, defaulting to 'not-a-knot'."
        bc_type_lower = "not-a-knot"
    end

    h = diff(x)
    if any(h .<= 0)
        throw(ArgumentError("x must be strictly increasing"))
    end

    a = y  # a[i] = y[i]

    # Set up tridiagonal system for second derivatives (c coefficients)
    n_eq = n - 2  # Number of equations
    A = zeros(Float64, n_eq, n_eq)
    B = zeros(Float64, n_eq)

    for i = 1:n_eq
        if i == 1  # First row: equation for c[2]
            if bc_type_lower == "not-a-knot"
                # Correct A and B for not-a-knot
                A[i, i] = (h[1] + h[2]) * (h[1] + 2*h[2]) / h[2]
                A[i, i + 1] = h[2] - h[1]^2 / h[2]
            else  # natural or clamped
                A[i, i] = 2 * (h[1] + h[2])
                if n_eq > 1
                    A[i, i + 1] = h[2]
                end
            end
        elseif i == n_eq  # Last row: equation for c[n-1]
            if bc_type_lower == "not-a-knot"
                # Correct A and B for not-a-knot
                A[i, i - 1] = h[n-2] - h[n-1]^2 / h[n-2]
                A[i, i] = (h[n-2] + h[n - 1]) * (h[n-1] + 2*h[n-2]) / h[n-2]
            else  # natural or clamped
                A[i, i] = 2 * (h[n - 2] + h[n - 1])
                A[i, i - 1] = h[n - 2]
            end
        else  # Interior rows: equation for c[i+1]
            A[i, i - 1] = h[i]
            A[i, i] = 2 * (h[i] + h[i + 1])
            A[i, i + 1] = h[i + 1]
        end

        # Construct B vector
        B[i] = 3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])
    end

    # Solve the tridiagonal system for c_interior
    c_interior = A \ B

    c = zeros(Float64, n)
    c[2:n - 1] .= c_interior

    # Calculate c[1] and c[n] based on the not-a-knot conditions
    if bc_type_lower == "not-a-knot"
        c[1] = ((h[1] + h[2]) / h[2]) * c[2] - (h[1] / h[2]) * c[3]
        c[n] = ((h[n - 2] + h[n - 1]) / h[n - 2]) * c[n - 1] - (h[n - 1] / h[n - 2]) * c[n - 2]
    elseif bc_type_lower == "natural" || bc_type_lower == "clamped"
        c[1] = 0.0
        c[n] = 0.0
    end

    # Calculate b and d coefficients
    b = zeros(Float64, n - 1)
    d = zeros(Float64, n - 1)
    for i = 1:n - 1
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    end

    return CubicSplineInterpolation(x, y, a, b, c, d, bc_type_lower, extrapolate)
end



"""
    evaluate_spline(spline::CubicSplineInterpolation, new_x::Vector{Float64})

Evaluates the cubic spline interpolation at new x values.

# Arguments
- `spline`: A CubicSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.

# Returns
- Array of interpolated y-values at the new x-coordinates.
"""
function evaluate_spline(spline::CubicSplineInterpolation, new_x::Vector{Float64})
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation
            if spline.extrapolate
                # Cubic Extrapolation
                if x_val < spline.x[1]
                    dx = x_val - spline.x[1]
                    idx = 1 # Use coefficients of the first interval
                    results[i] = spline.a[idx] + spline.b[idx] * dx + spline.c[idx] * dx^2 + spline.d[idx] * dx^3
                else # x_val > spline.x[end]
                    dx = x_val - spline.x[end-1] # Use coefficients of the last interval (index n-1)
                    idx = n - 1
                    results[i] = spline.a[idx] + spline.b[idx] * dx + spline.c[idx] * dx^2 + spline.d[idx] * dx^3
                end
            else
                results[i] = NaN
            end
        else
            # interior point -> Interpolation
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1
            elseif idx == n
                idx = n - 1
            end
            h = spline.x[idx+1] - spline.x[idx]
            dx = x_val - spline.x[idx]
            results[i] = spline.a[idx] + spline.b[idx] * dx + spline.c[idx] * dx^2 + spline.d[idx] * dx^3
        end
    end
    return results
end



"""
    evaluate_spline_derivative(spline::CubicSplineInterpolation, new_x::Vector{Float64})

Evaluates the derivative of the cubic spline interpolation at new x values,
consistent with cubic extrapolation when extrapolate=true.

# Arguments
- `spline`: A CubicSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative of the spline.

# Returns
- Array of derivative values at the new x-coordinates.
"""
function evaluate_spline_derivative(spline::CubicSplineInterpolation, new_x::Vector{Float64})
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            if spline.extrapolate
                # Derivative of Cubic Extrapolation
                if x_val < spline.x[1]
                    dx = x_val - spline.x[1]
                    idx = 1 # Use coefficients of the first interval for extrapolation to the left
                    results[i] = spline.b[idx] + 2 * spline.c[idx] * dx + 3 * spline.d[idx] * dx^2
                else # x_val > spline.x[end]
                    dx = x_val - spline.x[end-1] # Use coefficients of the last interval for extrapolation to the right
                    idx = n - 1
                    results[i] = spline.b[idx] + 2 * spline.c[idx] * dx + 3 * spline.d[idx] * dx^2
                end
            else
                results[i] = NaN # Or throw an error, or return a default value
            end
        else
            # Interpolation range: Derivative is calculated as before
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0 # x_val is exactly spline.x[1]
                idx = 1
            elseif idx == n # x_val is exactly spline.x[end] - should not happen because of < and > condition
                idx = n - 1 # Fallback to last interval
            end

            dx = x_val - spline.x[idx]
            results[i] = spline.b[idx] + 2 * spline.c[idx] * dx + 3 * spline.d[idx] * dx^2
        end
    end
    return results
end



"""
    evaluate_spline_antiderivative(spline::CubicSplineInterpolation, new_x::Vector{Float64}; C::Float64=0.0)

Evaluates the antiderivative (indefinite integral) of the cubic spline.

# Arguments
- `spline`: A `CubicSplineInterpolation` object.
- `new_x`:  A vector of x-values at which to evaluate the antiderivative.
- `C`: The constant of integration. Defaults to the negative of the antiderivative evaluated at spline.x[1].

# Returns
- A vector of the antiderivative values at the corresponding `new_x` points.

# Notes
The antiderivative is calculated interval by interval. The constant of integration
`C` is added to all results. The default value for `C` is chosen such that the
antiderivative is zero at the *first* knot point (`spline.x[1]`). Extrapolation
behavior is controlled by the `spline.extrapolate` setting.

"""
function evaluate_spline_antiderivative(spline::CubicSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    # Pre-calculate the cumulative integral values at the knots.
    # These act as the integration constants for each segment.
    cumulative_integrals = zeros(Float64, n)
    for i in 2:n
        h = spline.x[i] - spline.x[i-1]
        cumulative_integrals[i] = cumulative_integrals[i-1] +
                                 spline.a[i-1] * h +
                                 spline.b[i-1] * h^2 / 2 +
                                 spline.c[i-1] * h^3 / 3 +
                                 spline.d[i-1] * h^4 / 4
    end

    # Determine default C if not provided
    if isnan(C)
       C = 0.0 # Default: antiderivative is 0 at the first knot.
    end

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            if spline.extrapolate
                # Extrapolation.
                if x_val < spline.x[1]
                    dx = x_val - spline.x[1]
                    idx = 1  # Use first interval coefficients.
                    results[i] = C + spline.a[idx] * dx + spline.b[idx] * dx^2 / 2 + spline.c[idx] * dx^3 / 3 + spline.d[idx] * dx^4 / 4
                else  # x_val > spline.x[end]
                    dx = x_val - spline.x[n-1]  # Use *last* interval (n-1).
                    idx = n - 1
                    results[i] = C + cumulative_integrals[n-1] + spline.a[idx] * dx + spline.b[idx] * dx^2 / 2 + spline.c[idx] * dx^3 / 3 + spline.d[idx] * dx^4 / 4
                end
            else
                results[i] = NaN  # Outside the domain, return NaN.
            end
        else
            # Interpolation: Find the correct interval.
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1
            elseif idx == n
                idx = n - 1
            end
            dx = x_val - spline.x[idx]
            # Evaluate the antiderivative for the current interval.
            results[i] = C + cumulative_integrals[idx] + spline.a[idx] * dx + spline.b[idx] * dx^2 / 2 + spline.c[idx] * dx^3 / 3 + spline.d[idx] * dx^4 / 4
        end
    end
    return results
end



function safe_spline(x, y, x_new)
    # Simple cubic spline interpolation function that works with dual numbers
    # for automatic differentiation via ForwardDiff
    
    # Handle single query point case
    if !(x_new isa AbstractArray)
        return safe_spline(x, y, [x_new])[1]
    end
    
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n < 3
        throw(ArgumentError("At least 3 data points are required for cubic spline interpolation"))
    end
    
    # Calculate step sizes
    h = diff(x)
    
    # Set up tridiagonal system for second derivatives
    n_eq = n - 2
    A = zeros(promote_type(eltype(x), eltype(y)), n_eq, n_eq)
    B = zeros(promote_type(eltype(x), eltype(y)), n_eq)
    
    # Fill tridiagonal matrix using not-a-knot boundary conditions
    for i = 1:n_eq
        if i == 1
            # First row: equation for c[2]
            A[i, i] = (h[1] + h[2]) * (h[1] + 2*h[2]) / h[2]
            if n_eq > 1
                A[i, i + 1] = h[2] - h[1]^2 / h[2]
            end
        elseif i == n_eq
            # Last row: equation for c[n-1]
            A[i, i - 1] = h[n-2] - h[n-1]^2 / h[n-2]
            A[i, i] = (h[n-2] + h[n-1]) * (h[n-1] + 2*h[n-2]) / h[n-2]
        else
            # Interior rows
            A[i, i - 1] = h[i]
            A[i, i] = 2 * (h[i] + h[i + 1])
            A[i, i + 1] = h[i + 1]
        end
        
        # Construct right-hand side
        B[i] = 3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])
    end
    
    # Solve for interior second derivatives
    c_interior = A \ B
    
    # Initialize all second derivatives
    c = zeros(promote_type(eltype(x), eltype(y)), n)
    c[2:n-1] .= c_interior
    
    # Apply not-a-knot boundary conditions for first and last points
    c[1] = ((h[1] + h[2]) / h[2]) * c[2] - (h[1] / h[2]) * c[3]
    c[n] = ((h[n-2] + h[n-1]) / h[n-2]) * c[n-1] - (h[n-1] / h[n-2]) * c[n-2]
    
    # Calculate other coefficients for each segment
    b = zeros(promote_type(eltype(x), eltype(y)), n - 1)
    d = zeros(promote_type(eltype(x), eltype(y)), n - 1)
    for i = 1:n-1
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    end
    
    # Evaluate at new points
    results = zeros(promote_type(eltype(x), eltype(y), eltype(x_new)), length(x_new))
    
    for (j, xj) in enumerate(x_new)
        if xj <= x[1]
            # Extrapolate before first point
            dx = xj - x[1]
            results[j] = y[1] + b[1] * dx + c[1] * dx^2 + d[1] * dx^3
        elseif xj >= x[end]
            # Extrapolate after last point
            dx = xj - x[n-1]
            results[j] = y[n-1] + b[n-1] * dx + c[n-1] * dx^2 + d[n-1] * dx^3
        else
            # Interpolate: find segment using linear search (for dual number compatibility)
            i = 1
            # Using linear search instead of binary search for dual number compatibility
            while i < n && x[i+1] <= xj
                i += 1
            end
            
            if i >= n
                i = n - 1
            end
            
            dx = xj - x[i]
            results[j] = y[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
        end
    end
    
    return results
end



# --- Linear Spline Implementation ---

struct LinearSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    slopes::Vector{Float64}
    extrapolate::Bool
end

"""
    LinearSpline(x, y; extrapolate=true)

Constructs a LinearSplineInterpolation object from given x and y data.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true.

# Returns
- A LinearSplineInterpolation object.
"""
function LinearSpline(x::Vector{Float64}, y::Vector{Float64}; extrapolate::Bool=true)
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n < 2
        throw(ArgumentError("At least 2 data points are required for linear spline interpolation"))
    end
    if !issorted(x)
        throw(ArgumentError("x must be strictly increasing"))
    end

    slopes = diff(y) ./ diff(x)
    return LinearSplineInterpolation(x, y, slopes, extrapolate)
end

"""
    evaluate_spline(spline::LinearSplineInterpolation, new_x::Vector{Float64})

Evaluates the linear spline interpolation at new x values.

# Arguments
- `spline`: A LinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.

# Returns
- Array of interpolated y-values at the new x-coordinates.
"""
function evaluate_spline(spline::LinearSplineInterpolation, new_x::Vector{Float64})
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation
            if spline.extrapolate
                if x_val < spline.x[1]
                    # Use the first segment for extrapolation
                    idx = 1
                    results[i] = spline.y[idx] + spline.slopes[idx] * (x_val - spline.x[idx])
                else
                    # Use the last segment for extrapolation
                    idx = n - 1
                    results[i] = spline.y[idx] + spline.slopes[idx] * (x_val - spline.x[idx])
                end
            else
                results[i] = NaN
            end
        else
            # Interpolation
            idx = searchsortedfirst(spline.x, x_val) - 1
             if idx == 0
                idx = 1  # x is before first data point
            elseif idx == n
                idx = n-1
            end
            results[i] = spline.y[idx] + spline.slopes[idx] * (x_val - spline.x[idx])
        end
    end
    return results
end

"""
    evaluate_spline_derivative(spline::LinearSplineInterpolation, new_x::Vector{Float64})

Evaluates the derivative of the linear spline.  The derivative is piecewise
constant.

# Arguments
- `spline`: A LinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative.

# Returns
- Array of derivative values at the new x-coordinates.
"""
function evaluate_spline_derivative(spline::LinearSplineInterpolation, new_x::Vector{Float64})
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation: use slope of first/last segment
            if spline.extrapolate
                if x_val < spline.x[1]
                    results[i] = spline.slopes[1]
                else
                    results[i] = spline.slopes[end]
                end
            else
                results[i] = NaN  # Indicate extrapolation is not allowed
            end
        else
            # Interpolation: find correct segment
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1 #should not happen in theory, but added as defense
            elseif idx == n
                idx = n-1 #should not happen in theory, but added as defense
            end
            results[i] = spline.slopes[idx]
        end
    end
    return results
end


"""
    evaluate_spline_antiderivative(spline::LinearSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)

Evaluates the antiderivative (indefinite integral) of the linear spline.

# Arguments
- `spline`: A `LinearSplineInterpolation` object.
- `new_x`: A vector of x-values at which to evaluate the antiderivative.
- `C`: The constant of integration.  Defaults such that the
      antiderivative is zero at the *first* knot (`spline.x[1]`).

# Returns
- A vector of the antiderivative values.
"""
function evaluate_spline_antiderivative(spline::LinearSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    # Pre-calculate cumulative integrals at knot points.
    cumulative_integrals = zeros(Float64, n)
    for i in 2:n
        h = spline.x[i] - spline.x[i-1]
        cumulative_integrals[i] = cumulative_integrals[i-1] + spline.y[i-1] * h + 0.5 * spline.slopes[i-1] * h^2
    end

    # Determine default C if not provided
    if isnan(C)
        C = 0.0  # Default: antiderivative is 0 at the first knot
    end

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            if spline.extrapolate
                # Extrapolation
                if x_val < spline.x[1]
                    dx = x_val - spline.x[1]
                    idx = 1
                    results[i] = C + spline.y[idx] * dx + 0.5 * spline.slopes[idx] * dx^2
                else  # x_val > spline.x[end]
                    dx = x_val - spline.x[n]  # Corrected: Use spline.x[n], not spline.x[n-1]
                    idx = n - 1
                    # Corrected: Add y[n] * (x_val - spline.x[n])
                    results[i] = C + cumulative_integrals[n] + spline.y[n] * dx + 0.5 * spline.slopes[idx] * dx^2
                end
            else
                results[i] = NaN
            end
        else
            # Interpolation
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1
            elseif idx == n
                idx = n - 1
            end
            dx = x_val - spline.x[idx]
            results[i] = C + cumulative_integrals[idx] + spline.y[idx] * dx + 0.5 * spline.slopes[idx] * dx^2
        end
    end
    return results
end




# --- Pchip Spline Implementation ---

# Define the Spline object to store spline information
struct PchipSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    m::Vector{Float64}  # Slopes at each point
    a::Vector{Float64}  # a[i] = y[i]
    b::Vector{Float64}  # Coefficients for (x-x[i])
    c::Vector{Float64}  # Coefficients for (x-x[i])^2
    d::Vector{Float64}  # Coefficients for (x-x[i])^3
    extrapolate::Bool
end

"""
    PchipSpline(x, y; extrapolate=true)

Constructs a PchipSplineInterpolation object from given x and y data.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true.

# Returns
- A PchipSplineInterpolation object.
"""
function PchipSpline(x::Vector{Float64}, y::Vector{Float64}; extrapolate::Bool=true)
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n < 2
        throw(ArgumentError("At least 2 data points are required for PCHIP interpolation"))
    end
    if !issorted(x)
        throw(ArgumentError("x must be strictly increasing"))
    end

    h = diff(x)
    delta = diff(y) ./ h

    # Compute slopes (m) using Fritsch-Carlson method
    m = zeros(Float64, n)

    # Interior points
    for i = 2:n-1
        if sign(delta[i-1]) != sign(delta[i])
            m[i] = 0.0
        else
            w1 = 2 * h[i] + h[i-1]
            w2 = h[i] + 2 * h[i-1]
            m[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
        end
    end

    # Endpoint slopes (special handling to ensure monotonicity and shape preservation)
    m[1] = ((2 * h[1] + h[2]) * delta[1] - h[1] * delta[2]) / (h[1] + h[2])
    if sign(m[1]) != sign(delta[1])
      m[1] = 0.0
    elseif sign(delta[1]) != sign(delta[2]) && abs(m[1]) > abs(3*delta[1])
        m[1] = 3*delta[1]
    end
    
    m[n] = ((2 * h[n-1] + h[n-2]) * delta[n-1] - h[n-1] * delta[n-2]) / (h[n-1] + h[n-2])
    if sign(m[n]) != sign(delta[n-1])
        m[n] = 0.0
    elseif sign(delta[n-1]) != sign(delta[n-2]) && abs(m[n]) > abs(3 * delta[n-1])
      m[n] = 3*delta[n-1]
    end


    # Calculate coefficients
    a = y
    b = zeros(Float64, n - 1)
    c = zeros(Float64, n - 1)
    d = zeros(Float64, n - 1)

    for i = 1:n-1
        b[i] = m[i]
        c[i] = (3 * delta[i] - 2 * m[i] - m[i+1]) / h[i]
        d[i] = (m[i] + m[i+1] - 2 * delta[i]) / h[i]^2
    end

    return PchipSplineInterpolation(x, y, m, a, b, c, d, extrapolate)
end

"""
    evaluate_spline(spline::PchipSplineInterpolation, new_x::Vector{Float64})

Evaluates the PCHIP spline interpolation at new x values.

# Arguments
- `spline`: A PchipSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.

# Returns
- Array of interpolated y-values at the new x-coordinates.
"""
function evaluate_spline(spline::PchipSplineInterpolation, new_x::Vector{Float64})
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation (linear, using the slope at the endpoint)
            if spline.extrapolate
                if x_val < spline.x[1]
                    dx = x_val - spline.x[1]
                    results[i] = spline.a[1] + spline.m[1] * dx  # Use slope m[1]
                else
                    dx = x_val - spline.x[n]
                    results[i] = spline.a[n] + spline.m[n] * dx  # Use slope m[n]
                end
            else
                results[i] = NaN
            end
        else
            # Interpolation
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1
            elseif idx == n
                idx = n - 1
            end
            dx = x_val - spline.x[idx]
            results[i] = spline.a[idx] + spline.b[idx] * dx + spline.c[idx] * dx^2 + spline.d[idx] * dx^3
        end
    end
    return results
end


"""
    evaluate_spline_derivative(spline::PchipSplineInterpolation, new_x::Vector{Float64})

Evaluates the derivative of the PCHIP spline interpolation at new x values.

# Arguments
- `spline`: A PchipSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative.

# Returns
- Array of derivative values at the new x-coordinates.
"""
function evaluate_spline_derivative(spline::PchipSplineInterpolation, new_x::Vector{Float64})
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    for (i, x_val) in enumerate(new_x)
       if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation (constant, using the slope at the endpoint)
            if spline.extrapolate
                if x_val < spline.x[1]
                    results[i] = spline.m[1]
                else
                    results[i] = spline.m[n]
                end
            else
                results[i] = NaN
            end
        else
            # Interpolation
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1
            elseif idx == n
                idx = n - 1
            end
            dx = x_val - spline.x[idx]
            results[i] = spline.b[idx] + 2 * spline.c[idx] * dx + 3 * spline.d[idx] * dx^2
        end
    end
    return results
end


"""
    evaluate_spline_antiderivative(spline::PchipSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)

Evaluates the antiderivative (indefinite integral) of the PCHIP spline.

# Arguments
- `spline`: A `PchipSplineInterpolation` object.
- `new_x`: A vector of x-values at which to evaluate the antiderivative.
- `C`:  Constant of integration. Defaults to make the antiderivative zero at x[1].

# Returns
- A vector of the antiderivative values at the corresponding `new_x` points.

# Notes
Like with the cubic spline, the integral is computed piecewise, using a
pre-calculated cumulative integral at the knots.
"""
function evaluate_spline_antiderivative(spline::PchipSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)
    n = length(spline.x)
    results = zeros(Float64, length(new_x))

    # Pre-calculate cumulative integrals at knot points.
    cumulative_integrals = zeros(Float64, n)
    for i in 2:n
        h = spline.x[i] - spline.x[i-1]
        cumulative_integrals[i] = cumulative_integrals[i-1] +
                                 spline.a[i-1] * h +
                                 spline.b[i-1] * h^2 / 2 +
                                 spline.c[i-1] * h^3 / 3 +
                                 spline.d[i-1] * h^4 / 4
    end

     # Determine default C if not provided
    if isnan(C)
        C = 0.0  # Default: antiderivative is 0 at the first knot
    end

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            if spline.extrapolate
                # Extrapolation.  Use slope at the endpoint.
                if x_val < spline.x[1]
                    dx = x_val - spline.x[1]
                    results[i] = C + spline.y[1] * dx + 0.5 * spline.m[1] * dx^2 # Linear extrapolation
                else  # x_val > spline.x[end]
                    dx = x_val - spline.x[n]
                    results[i] = C + cumulative_integrals[n] + spline.y[n] * dx + 0.5*spline.m[n] * dx^2  # Linear extrapolation
                end
            else
                results[i] = NaN  # Outside the domain, return NaN.
            end
        else
            # Interpolation: Find the correct interval.
            idx = searchsortedfirst(spline.x, x_val) - 1
            if idx == 0
                idx = 1
            elseif idx == n
                idx = n - 1
            end
            dx = x_val - spline.x[idx]
            # Evaluate the antiderivative for the current interval.
            results[i] = C + cumulative_integrals[idx] + spline.a[idx] * dx + spline.b[idx] * dx^2 / 2 + spline.c[idx] * dx^3 / 3 + spline.d[idx] * dx^4 / 4
        end
    end
    return results
end


# Pchip safe for automatic differentiation
function safe_pchip(x, y, x_new)
    # PCHIP interpolation function that works with dual numbers
    # for automatic differentiation via ForwardDiff
    
    # Handle single query point case
    if !(x_new isa AbstractArray)
        return safe_pchip(x, y, [x_new])[1]
    end
    
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n < 2
        throw(ArgumentError("At least 2 data points are required for PCHIP interpolation"))
    end
    
    # Calculate step sizes and slopes
    h = diff(x)
    delta = diff(y) ./ h
    
    # Compute slopes (m) using Fritsch-Carlson method
    # Initialize with correct type for dual number compatibility
    m = zeros(promote_type(eltype(x), eltype(y)), n)
    
    # Interior points
    for i = 2:n-1
        if delta[i-1] * delta[i] <= 0  # Different signs or one is zero
            m[i] = 0.0
        else
            w1 = 2 * h[i] + h[i-1]
            w2 = h[i] + 2 * h[i-1]
            m[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
        end
    end
    
    # Endpoint slopes with monotonicity preservation
    if n > 2
        # First point
        m[1] = ((2 * h[1] + h[2]) * delta[1] - h[1] * delta[2]) / (h[1] + h[2])
        if delta[1] * m[1] <= 0  # Different signs or slope is zero
            m[1] = 0.0
        elseif (delta[1] * delta[2] <= 0) && (abs(m[1]) > abs(3 * delta[1]))
            m[1] = 3 * delta[1]
        end
        
        # Last point
        m[n] = ((2 * h[n-1] + h[n-2]) * delta[n-1] - h[n-1] * delta[n-2]) / (h[n-1] + h[n-2])
        if delta[n-1] * m[n] <= 0  # Different signs or slope is zero
            m[n] = 0.0
        elseif (delta[n-1] * delta[n-2] <= 0) && (abs(m[n]) > abs(3 * delta[n-1]))
            m[n] = 3 * delta[n-1]
        end
    else
        # Special case for n=2: use simple secant slopes
        m[1] = delta[1]
        m[n] = delta[1]
    end
    
    # Calculate polynomial coefficients for each segment
    T = promote_type(eltype(x), eltype(y))
    b = zeros(T, n - 1)
    c = zeros(T, n - 1)
    d = zeros(T, n - 1)
    
    for i = 1:n-1
        b[i] = m[i]
        c[i] = (3 * delta[i] - 2 * m[i] - m[i+1]) / h[i]
        d[i] = (m[i] + m[i+1] - 2 * delta[i]) / (h[i] * h[i])
    end
    
    # Evaluate at new points
    results = zeros(promote_type(eltype(x), eltype(y), eltype(x_new)), length(x_new))
    
    for (j, xj) in enumerate(x_new)
        if xj <= x[1]
            # Extrapolate before first point (linear extrapolation using the first slope)
            dx = xj - x[1]
            results[j] = y[1] + m[1] * dx
        elseif xj >= x[end]
            # Extrapolate after last point (linear extrapolation using the last slope)
            dx = xj - x[n]
            results[j] = y[n] + m[n] * dx
        else
            # Interpolate: find segment using linear search (for dual number compatibility)
            i = 1
            while i < n && x[i+1] < xj
                i += 1
            end
            
            if i >= n
                i = n - 1
            end
            
            dx = xj - x[i]
            results[j] = y[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
        end
    end
    
    return results
end








# --- Bilinear Spline Implementation ---

struct BilinearSplineInterpolation
    x::Vector{Float64}  # x grid points
    y::Vector{Float64}  # y grid points
    z::Matrix{Float64}  # z values at grid points z[i,j] = f(x[i], y[j])
    # Precomputed coefficients for each cell
    # For a cell defined by (x[i], y[j]), (x[i+1], y[j+1]), we store:
    a::Matrix{Float64}  # a[i,j] = z[i,j] (value at lower left corner)
    b::Matrix{Float64}  # b[i,j] = coefficient for (x-x[i])
    c::Matrix{Float64}  # c[i,j] = coefficient for (y-y[j])
    d::Matrix{Float64}  # d[i,j] = coefficient for (x-x[i])*(y-y[j])
    extrapolate::Bool   # Whether to extrapolate for points outside the grid
    bc_type::String     # Boundary condition type: "linear" or "constant"
end

"""
    BilinearSpline(x, y, z; extrapolate=true, bc_type="linear")

Constructs a BilinearSplineInterpolation object from given x, y grid and z values.

# Arguments
- `x`: Array of x-coordinates forming a grid. Must be strictly increasing.
- `y`: Array of y-coordinates forming a grid. Must be strictly increasing.
- `z`: Matrix of z-values at the grid points, where z[i,j] corresponds to f(x[i], y[j]).
       Must have dimensions length(x) × length(y).
- `extrapolate`: Boolean indicating whether to extrapolate for points outside the
                 range of the input grid. Defaults to true.
- `bc_type`: String indicating the boundary condition type for extrapolation.
             Options are:
             - "linear": Linear extrapolation (continues the interpolating function)
             - "constant": Constant extrapolation (uses the nearest edge value)
             Defaults to "linear".

# Returns
- A BilinearSplineInterpolation object.
"""
function BilinearSpline(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64}; 
                        extrapolate::Bool=true, bc_type::String="linear")
    nx = length(x)
    ny = length(y)
    
    if size(z) != (nx, ny)
        throw(DimensionMismatch("z matrix dimensions must match the length of x and y: expected size $(nx) × $(ny), got $(size(z))"))
    end
    
    if nx < 2 || ny < 2
        throw(ArgumentError("At least 2 data points in each dimension are required for bilinear interpolation"))
    end
    
    if !issorted(x) || !issorted(y)
        throw(ArgumentError("x and y must be strictly increasing"))
    end
    
    if !(bc_type in ["linear", "constant"])
        throw(ArgumentError("bc_type must be one of 'linear' or 'constant'"))
    end
    
    # Precompute coefficients for each cell in the grid
    a = zeros(Float64, nx-1, ny-1)
    b = zeros(Float64, nx-1, ny-1)
    c = zeros(Float64, nx-1, ny-1)
    d = zeros(Float64, nx-1, ny-1)
    
    for i in 1:(nx-1)
        for j in 1:(ny-1)
            # Cell corner values
            z11 = z[i, j]
            z21 = z[i+1, j]
            z12 = z[i, j+1]
            z22 = z[i+1, j+1]
            
            # Cell dimensions
            dx = x[i+1] - x[i]
            dy = y[j+1] - y[j]
            
            # Bilinear interpolation coefficients
            a[i, j] = z11
            b[i, j] = (z21 - z11) / dx
            c[i, j] = (z12 - z11) / dy
            d[i, j] = (z22 - z21 - z12 + z11) / (dx * dy)
        end
    end
    
    return BilinearSplineInterpolation(x, y, z, a, b, c, d, extrapolate, bc_type)
end

"""
    evaluate_spline(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})

Evaluates the bilinear spline interpolation at new (x,y) coordinates.

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.
- `new_y`: Array of new y-coordinates at which to evaluate the spline.
               Must have the same length as new_x.

# Returns
- Array of interpolated z-values at the new coordinates.
"""
function evaluate_spline(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})
    if length(new_x) != length(new_y)
        throw(DimensionMismatch("new_x and new_y must have the same length"))
    end
    
    results = zeros(Float64, length(new_x))
    
    for i in 1:length(new_x)
        x_val = new_x[i]
        y_val = new_y[i]
        
        # Check if point is outside the grid
        outside_x = x_val < spline.x[1] || x_val > spline.x[end]
        outside_y = y_val < spline.y[1] || y_val > spline.y[end]
        
        if (outside_x || outside_y) && !spline.extrapolate
            results[i] = NaN
            continue
        elseif (outside_x || outside_y) && spline.extrapolate && spline.bc_type == "constant"
            # Use closest point on the boundary for constant extrapolation
            x_clamped = clamp(x_val, spline.x[1], spline.x[end])
            y_clamped = clamp(y_val, spline.y[1], spline.y[end])
            results[i] = evaluate_point(spline, x_clamped, y_clamped)
            continue
        end
        
        # For linear extrapolation or points inside the grid
        results[i] = evaluate_point(spline, x_val, y_val)
    end
    
    return results
end

"""
    evaluate_point(spline::BilinearSplineInterpolation, x_val::Float64, y_val::Float64)

Helper function that evaluates the bilinear spline at a single point (x_val, y_val).

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `x_val`: x-coordinate at which to evaluate the spline.
- `y_val`: y-coordinate at which to evaluate the spline.

# Returns
- Interpolated z-value at the given coordinates.
"""
function evaluate_point(spline::BilinearSplineInterpolation, x_val::Float64, y_val::Float64)
    # Find grid cell containing the point
    x_idx = searchsortedfirst(spline.x, x_val) - 1
    y_idx = searchsortedfirst(spline.y, y_val) - 1
    
    # Handle edge cases for indices
    if x_idx == 0
        x_idx = 1
    elseif x_idx >= length(spline.x)
        x_idx = length(spline.x) - 1
    end
    
    if y_idx == 0
        y_idx = 1
    elseif y_idx >= length(spline.y)
        y_idx = length(spline.y) - 1
    end
    
    # Get relative coordinates within the cell
    dx = x_val - spline.x[x_idx]
    dy = y_val - spline.y[y_idx]
    
    # Use precomputed coefficients for bilinear interpolation
    # f(x,y) = a + b*(x-x1) + c*(y-y1) + d*(x-x1)*(y-y1)
    return spline.a[x_idx, y_idx] + 
           spline.b[x_idx, y_idx] * dx + 
           spline.c[x_idx, y_idx] * dy + 
           spline.d[x_idx, y_idx] * dx * dy
end


# needs testing
"""
    evaluate_spline_derivative_x(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})

Evaluates the partial derivative with respect to x of the bilinear spline.

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative.
- `new_y`: Array of new y-coordinates at which to evaluate the derivative.

# Returns
- Array of partial derivative values with respect to x at the new coordinates.
"""
function evaluate_spline_derivative_x(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})
    if length(new_x) != length(new_y)
        throw(DimensionMismatch("new_x and new_y must have the same length"))
    end
    
    results = zeros(Float64, length(new_x))
    
    for i in 1:length(new_x)
        x_val = new_x[i]
        y_val = new_y[i]
        
        # Check if point is outside the grid
        outside_x = x_val < spline.x[1] || x_val > spline.x[end]
        outside_y = y_val < spline.y[1] || y_val > spline.y[end]
        
        if (outside_x || outside_y) && !spline.extrapolate
            results[i] = NaN
            continue
        elseif (outside_x || outside_y) && spline.extrapolate && spline.bc_type == "constant"
            # Derivative is zero for constant extrapolation in the direction of extrapolation
            if outside_x
                results[i] = 0.0
                continue
            end
            # For y outside but x inside, we still compute the derivative with clamped y
            y_clamped = clamp(y_val, spline.y[1], spline.y[end])
            y_val = y_clamped
        end
        
        # Find grid cell containing the point
        x_idx = searchsortedfirst(spline.x, x_val) - 1
        y_idx = searchsortedfirst(spline.y, y_val) - 1
        
        # Handle edge cases for indices
        if x_idx == 0
            x_idx = 1
        elseif x_idx >= length(spline.x)
            x_idx = length(spline.x) - 1
        end
        
        if y_idx == 0
            y_idx = 1
        elseif y_idx >= length(spline.y)
            y_idx = length(spline.y) - 1
        end
        
        # Get relative coordinate in y direction
        dy = y_val - spline.y[y_idx]
        
        # For bilinear f(x,y) = a + b*(x-x1) + c*(y-y1) + d*(x-x1)*(y-y1)
        # The x-derivative is: ∂f/∂x = b + d*(y-y1)
        results[i] = spline.b[x_idx, y_idx] + spline.d[x_idx, y_idx] * dy
    end
    
    return results
end

"""
    evaluate_spline_derivative_y(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})

Evaluates the partial derivative with respect to y of the bilinear spline.

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative.
- `new_y`: Array of new y-coordinates at which to evaluate the derivative.

# Returns
- Array of partial derivative values with respect to y at the new coordinates.
"""
function evaluate_spline_derivative_y(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})
    if length(new_x) != length(new_y)
        throw(DimensionMismatch("new_x and new_y must have the same length"))
    end
    
    results = zeros(Float64, length(new_x))
    
    for i in 1:length(new_x)
        x_val = new_x[i]
        y_val = new_y[i]
        
        # Check if point is outside the grid
        outside_x = x_val < spline.x[1] || x_val > spline.x[end]
        outside_y = y_val < spline.y[1] || y_val > spline.y[end]
        
        if (outside_x || outside_y) && !spline.extrapolate
            results[i] = NaN
            continue
        elseif (outside_x || outside_y) && spline.extrapolate && spline.bc_type == "constant"
            # Derivative is zero for constant extrapolation in the direction of extrapolation
            if outside_y
                results[i] = 0.0
                continue
            end
            # For x outside but y inside, we still compute the derivative with clamped x
            x_clamped = clamp(x_val, spline.x[1], spline.x[end])
            x_val = x_clamped
        end
        
        # Find grid cell containing the point
        x_idx = searchsortedfirst(spline.x, x_val) - 1
        y_idx = searchsortedfirst(spline.y, y_val) - 1
        
        # Handle edge cases for indices
        if x_idx == 0
            x_idx = 1
        elseif x_idx >= length(spline.x)
            x_idx = length(spline.x) - 1
        end
        
        if y_idx == 0
            y_idx = 1
        elseif y_idx >= length(spline.y)
            y_idx = length(spline.y) - 1
        end
        
        # Get relative coordinate in x direction
        dx = x_val - spline.x[x_idx]
        
        # For bilinear f(x,y) = a + b*(x-x1) + c*(y-y1) + d*(x-x1)*(y-y1)
        # The y-derivative is: ∂f/∂y = c + d*(x-x1)
        results[i] = spline.c[x_idx, y_idx] + spline.d[x_idx, y_idx] * dx
    end
    
    return results
end



"""
    evaluate_spline_grid(spline::BilinearSplineInterpolation, grid_x::Vector{Float64}, grid_y::Vector{Float64})

Evaluates the bilinear spline on a 2D grid of points. Useful for visualization.

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `grid_x`: Array of x-coordinates forming a grid.
- `grid_y`: Array of y-coordinates forming a grid.

# Returns
- A matrix of z-values with dimensions length(grid_x) × length(grid_y),
  where result[i,j] is the interpolated value at (grid_x[i], grid_y[j]).
"""
function evaluate_spline_grid(spline::BilinearSplineInterpolation, grid_x::Vector{Float64}, grid_y::Vector{Float64})
    nx = length(grid_x)
    ny = length(grid_y)
    result = zeros(Float64, nx, ny)
    
    for i in 1:nx
        for j in 1:ny
            x_val = grid_x[i]
            y_val = grid_y[j]
            
            # Use the single-point evaluation function
            result[i, j] = evaluate_spline(spline, [x_val], [y_val])[1]
        end
    end
    
    return result
end







# --- Grid Tools ---

"""
    make_grid(x_min, x_max, n, density=1.0)

Constructs a grid of `n` points between `x_min` and `x_max` with a given density. Higher density
values will concentrate more points near the lower bound `x_min`.

# Arguments
- `x_min`: Lower bound of the grid.
- `x_max`: Upper bound of the grid.
- `n`: Number of grid points.
- `density`: Density of grid points. Defaults to 1.0.

# Returns
- A vector of grid points.
"""
function make_grid(x_min::Float64, x_max::Float64, n::Int; density::Float64=1.0)
    x = range(0, 1, length=n)
    x = x .^ density
    x = x_min .+ x * (x_max - x_min)
    return x
end





# --- Example Usage 1d ---

#=
x = collect(range(0.3, 5, 15))
f(x) = log(x)
y = f.(x)

# construct spline object and evaluate
spl = CubicSpline(x, y) #
x_new = collect(range(0.05, 8, 100))
y_spl = evaluate_spline(spl, x_new)
y_true = f.(x_new)

# evaluate derivatives of spline
f_prime(x) = 1 / x
y_prime_true = f_prime.(x_new)
y_prime_spl = evaluate_spline_derivative(spl, x_new)

# evaluate antiderivatives of the spline with constant of integration
f_int(x) = x * log(x) - x
y_int_true = f_int.(x_new)
C = f_int(x[end]) - evaluate_spline_antiderivative(spl, [x[end]])[1]
y_int_spl = evaluate_spline_antiderivative(spl, x_new) .+ C


# plot results
using Plots
plot(x_new, y_true, label="True", lw=2)
plot!(x_new, y_spl, label="Interpolated", lw=2)
scatter!(x, y, label="Spline Points")

# plot derivatives
plot(x_new, y_prime_true, label="True Derivative", lw=2)
plot!(x_new, y_prime_spl, label="Interpolated Derivative", lw=2)
scatter!(x, f_prime.(x), label="Spline Points")

# plot antiderivative
plot(x_new, y_int_true, label="True Antiderivative", lw=2)
plot!(x_new, y_int_spl, label="Interpolated Antiderivative", lw=2)
scatter!(x, f_int.(x), label="Spline Points")
=#




# --- Example Usage 2d ---
#=
function f(x, y)
    return x^(.5) * y^(.5)
end

# Create a sample 2D grid
x = collect(range(0.25, 4.0, 25))
y = collect(range(0.25, 4.0, 25))

# Create the z matrix (function values at grid points)
z = zeros(length(x), length(y))
for i in eachindex(x), j in eachindex(y)
    z[i, j] = f(x[i], y[j])
end

# Construct the bilinear spline with precomputed coefficients
bilinear_spline = BilinearSpline(x, y, z; bc_type="constant")

#Create a grid of points for visualization
grid_x = collect(range(0.0, 5.0, 100))
grid_y = collect(range(0.0, 5.0, 100))

# Create pairs for evaluation
eval_x = Float64[]
eval_y = Float64[]
for i in grid_x, j in grid_y
    push!(eval_x, i)
    push!(eval_y, j)
end

# Evaluate spline at grid points
grid_z = evaluate_spline(bilinear_spline, eval_x, eval_y)

# Reshape for visualization
grid_z_matrix = reshape(grid_z, (length(grid_x), length(grid_y)))

# Test derivatives at a specific point
test_x = [2.5]
test_y = [3.5]
test_z = evaluate_spline(bilinear_spline, test_x, test_y)
dx = evaluate_spline_derivative_x(bilinear_spline, test_x, test_y)
dy = evaluate_spline_derivative_y(bilinear_spline, test_x, test_y)

println("Interpolated value at ($(test_x[1]), $(test_y[1])): $(test_z[1])")
println("True value: $(f(test_x[1], test_y[1]))")
println("Derivative with respect to x: $(dx[1])")
println("Derivative with respect to y: $(dy[1])")

# Visualize
using Plots
surface(grid_x, grid_y, grid_z_matrix', xlabel="x", ylabel="y", zlabel="z", title="Bilinear Spline Interpolation")
=#

