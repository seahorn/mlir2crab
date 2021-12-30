# -*- Python -*-
import os
import sys
import re
import platform
import lit.util
import lit.formats

config.name = 'MLIR2CRAB'
config.suffixes = ['.mlir']
config.test_format = lit.formats.ShTest(execute_external=False)
# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)
config.substitutions.append(('%mlir2crab', os.path.join(config.mlir2crab_install,'bin','mlir2crab')))

