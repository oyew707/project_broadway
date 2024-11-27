"""
-------------------------------------------------------
Module for creating a logger objects
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""


# Imports
import logging
from typing import Literal, Optional

# Constants
logging.basicConfig(
    format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.WARNING
)
log_levels = {
    'info': logging.INFO,
    'warn': logging.WARN,
    'debug': logging.DEBUG,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def getlogger(name: str, loglevel: Optional[Literal['info', 'warn', 'debug', 'error', 'critical']])\
        -> logging.Logger:
    """
    -------------------------------------------------------
    Creates a logger with the specified name and log level
    -------------------------------------------------------
    Parameters:
       name - the name of the logger (str)
       loglevel - the log level of the logger ('info', 'warn', 'debug', 'error', 'critical' None)
    Returns:
       logger - the logger object (logging.Logger)
    -------------------------------------------------------
    """
    logger = logging.getLogger(name=name)
    # Lowercase for case-insensitivity
    logger.setLevel(log_levels.get(loglevel.lower(), logging.NOTSET))
    return logger