# %%
from typing import Tuple
import math
from probability import normal_cdf
from probability import inverse_normal_cdf

# %%
def normal_approximation_to_binomial(n : int , p : float) : 
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# %%
normal_probability_below = normal_cdf #probability the variable is below a threshold

# %%
def normal_probability_above(lo : float ,
                             mu : float = 0 ,
                             sigma : float = 1) -> float: 
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

# %%
def normal_probability_between(lo : float ,
                               hi : float ,
                               mu : float = 0 ,
                               sigma : float = 1) -> float :
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# %%
def normal_probability_outside(lo : float ,
                               hi : float ,
                               mu : float = 0 , 
                               sigma : float = 1) -> float :
    """The probability that an N(mu, sigma) is not between lo and hi."""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

# %%
def normal_upper_bound(probability : float ,
                       mu : float = 0 ,
                       sigma : float = 1) -> float :
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability , mu , sigma)

# %%
def normal_lower_bound(probability : float ,
                       mu : float = 0,
                       sigma : float = 1) -> float :
    """Returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability , mu , sigma)

# %%
def normal_two_sided_bounds(probability : float ,
                            mu : float = 0 ,
                            sigma : float = 1) -> Tuple[float, float] :
    """
    Returns the symmetric (about the mean) bounds
    that contain the specified probability
    """
    tail_probability = (1 - probability) / 2

    upper_bound = normal_lower_bound(tail_probability , mu , sigma) #should have tail probability above it

    lower_bound = normal_upper_bound(tail_probability , mu , sigma) #should have tail probability below it

    return lower_bound, upper_bound

# %%
def two_sided_p_value(x : float , mu : float = 0 , sigma : float = 1) -> float :
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu , sigma)?
    """
    if x >= mu :
        return 2 * normal_probability_above(x, mu, sigma) #tail is everything greater than x
    
    else:
        return 2 * normal_probability_below(x, mu, sigma) #tail is everything less than x

# %%
def B(alpha : float , beta : float) -> float :
    """A normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

# %%
def beta_pdf(x : float , alpha : float , beta : float) -> float :
    if x <= 0 or x >= 1:          
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha , beta)



