from typing import List, Union, Dict, Any
from scipy.stats import norm
import functions
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Formats for plotting
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.2e' % x))
fmt = mticker.FuncFormatter(g)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Formats for printing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _superScriptinate(number):
    return number.replace('0','⁰').replace('1','¹').replace('2','²').\
        replace('3','³').replace('4','⁴').replace('5','⁵').replace('6','⁶')\
        .replace('7','⁷').replace('8','⁸').replace('9','⁹').replace('-','⁻')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sci_notation(number:float, sig_fig:int=2,
                 max:float=np.float_power(10, -100)
                 ) -> str:
    """
    Returns a number in scientific notation with the lead numbers to  a
    specific significant number `sig_fig`
    
    Automatically truncates values if too small to print.
    """
    if number < max:
        number = max
    # getting string
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    try:
        a,b = ret_string.split("e")
        # removed leading "+" and strips leading zeros too.
        b = int(b)
        return a + "×10" + _superScriptinate(str(b))
    except ValueError or TypeError:
        return str(np.nan)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _format_estimates(point:float, se:Union[float,None]=None,
                      lower:Union[float, None]=None,
                      upper:Union[float, None]=None, alpha:float=0.05,
                      round:int=2, exp:bool=False
                      ) -> str:
    """
    Formats point estimates with confidence intervals
    
    Arguments
    ---------
    point, se, lower, upper : float
        Please supply either the `se` or (`lower` and `upper`) as floats.
    alpha : float, default 0.05
        The desired type 1 error rate
    round : integer, default 2
        The desired number of significant figures
    exp : boolean, default False
        Should the point estimates, lower and upper bounds be exponentiated
        with base `e`
    
    Returns
    -------
    format_string : str
        `point (lower; upper)` formatted string with appropriate rounding
    """
    # check input
    functions.is_type(round, int)
    functions.is_type(alpha, float)
    functions.is_type(point, float)
    if se == None and not ( isinstance(lower, float) and isinstance(upper, float) ):
            raise TypeError('Please supply either an `se`, or both  `lower` '
                            'and `upper`')
    if isinstance(se, float) and not ( lower == None and upper == None):
            raise TypeError('Please supply either an `se`, or both  `lower` '
                            'and `upper`')
    if isinstance(se, float) and isinstance(lower, float) and\
    isinstance(upper, float):
            warnings.warn('Ignoring `se`', SyntaxWarning)
    # calculate lower and upper bounds
    if (lower == None and upper == None):
        z = norm.ppf(1-alpha/2)
        lower = point - z*se
        upper = point + z*se
    if exp == True:
        point = np.exp(point)
        upper = np.exp(upper)
        lower = np.exp(lower)
    # format string
    r_format = '.{}f'.format(round)
    format_string = format(point, r_format) + ' (' + format(lower, r_format) + \
        '; ' + format(upper, r_format) + ')'
    return format_string

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
_format_estimates_vec = np.vectorize(_format_estimates)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _scrape_numerics(string:str, element=0) -> List[float]:
    '''
    Function to extract a decimal number from a pandas dataframe column
    (e.g., a `pd.Series`) containing string formatted lists.
    
    Parameters
    ----------
    string : str
        A string list with one or more elements.
    element : int, default 0
        The element number that should be extracted.
    
    Returns
    -------
    lst: list
        A list with a single element per row of the original pd.Series.
    
    Examples
    --------
    
    >>> data = pd.DataFrame({'col' : ['[1.2, 1.4]', '[2.4, 1.3]']})
    >>> [_scrape_numerics(r, element=1) for r in data.col]
    
    '''
    # ### input
    functions.is_type(element, int)
    # ### algorithm
    try:
        nstr = ''.join((ch if ch in '0123456789.+-e' else ' ') for ch in string)
        lst = [float(i) for i in nstr.split()][element]
    except TypeError:
        if np.isnan(string):
            lst = np.nan
        else:
            raise TypeError('Please check input type')
    return lst

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def string_list_to_column(data:pd.DataFrame,
                          map_dict:Dict[str, List[Any]],
                          ) -> pd.DataFrame:
    '''
    Takes the dictionary keys from `map_dict` to select columns from `data`
    and scrapes a single numeric value per row from lists with one or more
    elements that were morphed into strings.
    
    Parameters
    ----------
    data : pd.DataFrame,
    map_dict : dict,
        Dictionary with keys that map the `data` columns, and lists as values:
        [`new column name string`, index integer], where the index indicates
        the element from the original string-list to return.
    
    Returns
    -------
    data : pd.DataFrame,
        The data enriched with additional columns.
    
    Examples
    --------
    
    >>> data = pd.DataFrame({
    >>>             'col1' : ['[1.2, 1.4]', '[2.4, 1.3]'],
    >>>             'col2' : ['[-1.2, 1.3]', '[-0.4, -4.3]'],
    >>>             })
    >>> map_dict = {'col1' : ['col1_new', 0], 'col2': ['better', 1]}
    >>> data_mapped = string_list_to_column(data, map_dict)
    
    '''
    # ### input
    functions.is_type(data, pd.DataFrame)
    functions.is_type(map_dict, dict)
    # ### algorithm
    for idx, do_this in map_dict.items():
        # what do we want to do
        new_col = do_this[0]
        row_idx = do_this[1]
        # old column
        old_col = data[idx]
        # new data
        data[new_col] = [ _scrape_numerics(r, row_idx) for r in old_col ]
    # return
    return data

