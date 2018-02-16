# import all gumpy submodules that should be loaded automatically
import gumpy.classification
import gumpy.data
import gumpy.plot
import gumpy.signal
import gumpy.utils
import gumpy.features
import gumpy.split

# fetch into gumpy-scope so that users don't have to specify the entire
# namespace
from gumpy.classification import classify

# retrieve the gumpy version (required for package manager)
from gumpy.version import __version__
