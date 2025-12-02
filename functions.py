#!/usr/bin/env python3
##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Script information
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## Script name: functions.py
##
## Purpose of script: Collect useful functions in one script
##
## Author: M. van Vugt
## Inspired by functions of: A.F. Schmidt
## Date created: 06/02/2023
##
## Copyright (c) M. van Vugt, 2023
## Email: m.vanvugt-2@umcutrecht.nl
##
##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Imports
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from typing import Tuple, Union, Any, Type
from scipy.stats import chi2, beta, randint
import pandas as pd


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## merit_helper constants
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class InputValidationError(Exception):
    pass

def is_type(param: Any, types: Union[Tuple[Type], Type]) -> bool:
    """
    Checks if a given parameter matches any of the supplied types
    
    Parameters
    ----------
    param: object to test
    types: either a single type, or a tuple of types to test against.
    
    Returns
    -------
    True if the parameter is an instance of any of the given types.
    Raises InputValidationError otherwise.
    """
    if not isinstance(param, types):
        raise InputValidationError(f"Expected any of [{types}], got {type(param)}")
    return True


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## CI for proportions
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _beta_confidence(proportion, total_sample, alpha=0.05, integer=True):
    """
    Better confidence intervals for proportions, based on a classic
    contribution by:
    .. [1] Steven A Julious, "Two-sided confidence intervals for the single
    proportion: comparison of seven methods by Robert G. Newcombe, Statistics
    in Medicine 1998; 17:857-872"
    
    Parameters
    ----------
    proportion : float
        A proportion between 0 and 1.
    total_sample : int
        The total sample size the proportion was derived from.
    alpha : float, default 0.05
        A float between 0 and 1, representing the type 1 error rate. Is used
        to define the confidence interval coverage: (1-alpha)*100.
    integer : boolean, default True
        If the function should fail if the proportion times total_sample does
        not result in an integer number of events. Set to `False` to ignore
        the ValueError.
    
    Returns
    -------
    Unpacks the lower and upper bounds
    """
    
    # check input
    is_type(proportion, float)
    is_type(total_sample, int)
    # get the number of events
    no_events = proportion * total_sample
    if (not no_events.is_integer()) and (integer==True):
        raise ValueError('proportion * total_sample is not an integer, either \
correct input or set `integer` to `False`')
        
    # lower bound
    lb = 1-beta.ppf(1-alpha/2, total_sample - no_events + 1, no_events)
    # upper bound
    ub = beta.ppf(1-alpha/2, no_events + 1, total_sample - no_events)
    # return
    return lb, ub


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Add notes to appendix tables
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO allow for lists, adding multiple rows of notes
# TODO allow for note in only first column
def add_supnote(data:pd.DataFrame, note:Union[str, list],
        name:Union[str, list, bool], index:bool = True):
    """
    Add notes to an appendix table
    
    Parameters
    ----------
    data : pd.DataFrame,
        Dataframe to which note should be added
    note : str or list,
        The content of the note(s) to be added
    index : boolean, default True
        Indicates if the index is used and labelled
    name : str, list, or boolean,
        Name(s) of the note(s) or `False` if first columns should be the note,
        can only be `False` if `index = False`, if not `False`, should be as
        long as `note`
    
    Returns
    -------
    The table with the note
    """

    # # check input
    is_type(data, pd.DataFrame)
    is_type(note, (str, list))
    is_type(index, bool)
    is_type(name, (str, list, bool))

    # save columns
    cols = data.columns

    if isinstance(note, str):
        # add single note
        if index:
            data = pd.concat([data, pd.DataFrame({cols[0] : note},
                index = [name], )], axis = 0)
        else:
            if not name:
                # add note as first column
                name = 'note'
                data = pd.concat([data, pd.DataFrame({cols[0] : note},
                    index = [name], )], axis=0)
            else:
                data = pd.concat([data, pd.DataFrame({cols[0] : name,
                    cols[1] : note}, index = [name], )], axis=0)
        # make other columns empty
        data.loc[name, :] = data.loc[name, :].fillna('').copy()
    else:
        # add multiple notes
        for i in range(len(note)):
            if index:
                # check if names and notes are equal in length
                if len(note) != len(name):
                    raise TypeError('Length of note and name should be equal')
                namin = name[i]
                data = pd.concat([data, pd.DataFrame({cols[0] : note[i]},
                    index = [namin], )], axis = 0)
            else:
                if not name:
                    # add note as first column
                    namin = 'note_' + str(i)
                    data = pd.concat([data, pd.DataFrame({cols[0] : note[i]},
                        index = [namin], )], axis=0)
                else:
                    namin = name[i]
                    data = pd.concat([data, pd.DataFrame({cols[0] : namin,
                        cols[1] : note[i]}, index = [namin], )], axis=0)
            # make other columns empty
            data.loc[namin, :] = data.loc[namin, :].fillna('').copy()

    # finish
    return data

