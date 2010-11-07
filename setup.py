# Author: Anand Patil
# Date: 3 June 2010
# License: Gnu GPL
#####################

from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('survey_eval_example',parent_package=None,top_path=None)
config.packages = ["survey_eval_example"]
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))