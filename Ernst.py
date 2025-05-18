"""
Generates an Ernst Profile with a user-specified aspect ratio
and number of k-terms as described in Tyler J. Mason et al., “Design and Preliminary Experimental Results on a 
Uniform-Field Test Fixture for Power Flow Experiments,” IEEE Trans. Plasma Sci., 2025.
"""

from scipy.optimize import fsolve, minimize
import numpy as np
# import warnings
# warnings.filterwarnings('ignore', 'The iteration is not making good progress')


def f(k_terms, v, order):
    """Finds the derivative of f evaluated at u=0 from equation 11 to the appropriate order to solve the system of equations
    Args:
        k_terms: a list of the k-terms used in computing the profile
        v: the equipotential used to determine the profile surface
        order: the order of the derivative taken
    returns:
        out: the derivative of f evaluated at u=0
    """
    if order == 0:
        out = 1  # adds 1 if no derivative is taken
    else:
        out = 0
    for n, ki in enumerate(k_terms, start=1):
        out += n ** (1 + order) * ki * np.cos(n * v)
    return out


def g(k_terms, v, order):
    """Finds the derivative of g evaluated at u=0 from equation 11 to the appropriate order to solve the system of equations
    Args:
        k_terms: a list of the k-terms used in computing the profile
        v: the equipotential used to determine the profile surface
        order: the order of the derivative taken
    Returns:
        out: the derivative of g evaluated at u=0
    """
    out = 0
    for n, ki in enumerate(k_terms, start=1):
        out += n ** (1 + order) * ki * np.sin(n * v)
    return out


def system_eqns(variables, k1):
    """Creates the system of equations given by equation 13
    Args:
        variables: a list of the variables to be solved for. the 0th index represents v, while the ramaining
            indicies represent k2, k3, ...
        k1: the value of k1 used in generating the profile
    Returns:
        Equations: A list of the system of equations given by equation 13
    """
    if type(variables) == np.float64:
        v = variables
        k_terms = [k1]
    else:
        v = variables[0]
        k_terms = [k1]
        for k in variables[1:]:
            k_terms.append(k)
    eqn_1 = f(k_terms, v, 0) * f(k_terms, v, 2) + g(k_terms, v, 1) * g(k_terms, v, 1)
    equations = [eqn_1]
    if len(k_terms) > 1:
        eqn_2 = (
            f(k_terms, v, 0) * f(k_terms, v, 4)
            + 4 * g(k_terms, v, 1) * g(k_terms, v, 3)
            + 3 * f(k_terms, v, 2) * f(k_terms, v, 2)
        )
        equations.append(eqn_2)
    if len(k_terms) > 2:
        eqn_3 = (
            f(k_terms, v, 0) * f(k_terms, v, 6)
            + 6 * g(k_terms, v, 1) * g(k_terms, v, 5)
            + 15 * f(k_terms, v, 2) * f(k_terms, v, 4)
            + 10 * g(k_terms, v, 3) * g(k_terms, v, 3)
        )
        equations.append(eqn_3)
    if len(k_terms) > 3:
        eqn_4 = (
            f(k_terms, v, 0) * f(k_terms, v, 8)
            + 8 * g(k_terms, v, 1) * g(k_terms, v, 7)
            + 28 * f(k_terms, v, 2) * f(k_terms, v, 6)
            + 56 * g(k_terms, v, 3) * g(k_terms, v, 5)
            + 35 * f(k_terms, v, 4) * f(k_terms, v, 4)
        )
        equations.append(eqn_4)
    if len(k_terms) > 4:
        eqn_4 = (
            f(k_terms, v, 0) * f(k_terms, v, 10)
            + 10 * g(k_terms, v, 1) * g(k_terms, v, 9)
            + 45 * f(k_terms, v, 2) * f(k_terms, v, 8)
            + 120 * g(k_terms, v, 3) * g(k_terms, v, 7)
            + 210 * f(k_terms, v, 4) * f(k_terms, v, 6)
            + 126 * g(k_terms, v, 5) * g(k_terms, v, 5)
        )
        equations.append(eqn_4)
    return equations


def find_x_y(k_terms, v):
    """Finds the coordinates of the profile using equation 9 for a given v and list of k-terms
    Args:
        k_terms: A list of the k-terms used to generate the profile
        v: the equipotential used to determine the profile surface
    Returns:
        x_y_coords: A 2D numpy array containing the coordinates of the profile
    """
    if type(k_terms) == np.float64:
        k_terms = [k_terms]
    n_points = 1000000
    x_y_coords = np.empty((n_points, 2))
    for i in range(n_points):
        u = i * 0.02
        x_y_coords[i,0] = u
        x_y_coords[i,1] = v
        for n, ki in enumerate(k_terms, start=1):
            x_y_coords[i,0] += ki * np.sinh(n * u) * np.cos(n * v)
            x_y_coords[i,1] += ki * np.cosh(n * u) * np.sin(n * v)
        if x_y_coords[i,0] < 0 or x_y_coords[i,1] < x_y_coords[i-1,1]:
            return x_y_coords[:i,:]
    raise Exception('Failed to converge')
    


def generate_profile(k1, n_k_terms):
    """Solves the system of equations given by equation 13 to find v and the higher-order k-values and returns the coordinates calculated in find_x_y
    Args:
        k1: The first k-term
        n_k_terms: The user-defined number of k-terms to be used in computing the profile
    Returns:
        coords: A 2D numpy array containing the coordinates of the profile, calculated by find_x_y()
    """
    if type(k1) == np.ndarray:
        k1 = k1[0]
    opt_params_guess = [np.pi/2, k1**2/8, k1**3/90, k1**4/900, k1**5/9000] # First item is v, remainder are k2, k3, ...
    opt_params = fsolve(system_eqns, opt_params_guess[:n_k_terms], args=(k1,))
    v = opt_params[0]
    k_terms = [k1]
    for ki in opt_params[1:]:
        k_terms.append(ki)
    coords = find_x_y(k_terms, v)
    return coords



def reflect(coords):
    """Creates a full profile by reflecting the coordinates found in generate_profile across the y-axis
    Args:
        coords: The coordinates of 1/2 ernst profile
    Retruns:
        coords: The coordinated of a full Ernst Profile
    """
    coords_reflect = np.transpose(np.array((-coords[1:, 0], coords[1:, 1])))
    coords_reflect = np.flip(coords_reflect, 0)
    coords = np.append(coords_reflect, coords, axis=0)
    return coords


def loss_wrapper(k1, ar, n_k_terms):
    """Determines how close the aspect ratio is to the desired aspect ratio for a given k1
    Args:
        k1: The first k-term
        ar: The desired aspect ratio
        n_k_terms: The user-defined number of k-terms to be used in computing the profile
    Returns:
        The mean-squared error between the desired aspect ratio and the computed aspect ratio
    """
    coords = generate_profile(k1, n_k_terms)
    width = 2 * np.max(coords[:, 0])
    height = coords[0, 1]
    ar_iteration = width / height
    return (ar - ar_iteration) ** 2


def optimize(ar, n_k_terms):
    """Finds the optimal value for k1 to generate the desired aspect ratio
    Args:
        ar: The desired aspect ratio
        n_k_terms: The user-defined number of k-terms to be used in computing the profile
    Returns:
        k1: the optimal value for k1 that generates a profile of the desired aspect ratio
    """
    # Initial guess of k1. The plot of aspect ratio as a function of k1 roughly follows the function
    # ar(k) = 12*exp(-3*k^0.5) for ar<10. For larger aspect ratios, we choose a small value of k1.
    # This initial guess may need to be fiddled with to work properly for stubborn aspect ratios
    if ar < 10:
        k1_guess = ((np.log(ar) - np.log(12)) / 3) ** 2
    else:
        k1_guess = 1e-5
    k1 = minimize(loss_wrapper, k1_guess, args=(ar, n_k_terms)).x
    return k1


def find_E(k1, n_k_terms):
    """Finds the electric field across the surface of the electrode
    Args:
        k1: The first k-term
        n_k_terms: The user-defined number of k-terms to be used in computing the profile
    Returns:
    E: The electric field along the surface
    """
    if type(k1) == np.ndarray:
        k1 = k1[0]
    opt_params_guess = [np.pi/2, k1**2/8, k1**3/90, k1**4/900, k1**5/9000]
    opt_params = fsolve(system_eqns, opt_params_guess[:n_k_terms], args=(k1,))
    v = opt_params[0]
    k_terms = [k1]
    for ki in opt_params[1:]:
        k_terms.append(ki)
    u = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    f = 1
    g = 0
    for n, ki in enumerate(k_terms, start=1):
        f += n * ki * np.cos(n * v) * np.cosh(n * u)
        g += n * ki * np.sin(n * v) * np.sinh(n * u)
    E_minus2 = f**2 + g**2
    E = E_minus2 ** (-0.5)
    return E


def main(*args):
    """Finds the optimal profile and electric field on the profile for a user-inputted electrode width,
    height, and number of k-terms and saves them as a .txt file
    Args:
        w_in = The desired Electrode Width
        h_in = The desired distance between the electrode and ground plane
        n_k_terms = The number of k-terms to be used
    Returns:
        coords: The coordinates of the Ernst profile
        E: The electric field along the surface of the electrode
    """
    if not args:
        w_in = eval(input("Electrode Width: "))
        h_in = eval(input("Distance between electrode and ground plane: "))
        n_k_terms = eval(input("Number of k-terms: "))
    elif len(args) == 3:
        w_in = args[0]
        h_in = args[1]
        n_k_terms = args[2]
    else:
        raise Exception('improper length of args')
    if n_k_terms > 5:
        raise Exception('This program only supports up to 5 k terms')
    ar_in = w_in / h_in
    k1 = optimize(ar_in, n_k_terms)
    coords = generate_profile(k1, n_k_terms)
    coords = reflect(coords)
    h_out = np.min(coords[:, 1])
    coords *= h_in / h_out
    E = find_E(k1, n_k_terms)
    np.savetxt('coords.txt', coords)
    np.savetxt('Efield.txt', E)
    return coords, E


if __name__ == "__main__":
    main()
