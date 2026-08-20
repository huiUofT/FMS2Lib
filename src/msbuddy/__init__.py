"""This code was obtained from the original BUDDY implementation developed by
Xing et al.

Original repository:
https://github.com/HuanLab/BUDDY

No modifications were made to the original code.

The original copyright notice and license terms are retained.
"""

from .cand import assign_subformula
from .base import Formula, Spectrum, Adduct, MetaFeature
from .main import Msbuddy, MsbuddyConfig
from .utils import read_formula, form_arr_to_str, enumerate_subform_arr

__version__ = "0.3.12"
__author__ = "Shipei Xing"
__email__ = "philipxsp@hotmail.com"
__author_email__ = "philipxsp@hotmail.com"

__copyright__ = "Copyright 2024 Shipei Xing"
__license_expression__ = "Apache 2.0"

__url__ = "https://github.com/Philipbear/msbuddy"
__docs__ = "https://msbuddy.readthedocs.io/en/latest"

__title__ = "msbuddy"
