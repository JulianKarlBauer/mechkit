import sys

if sys.version_info > (3, 0):
    from ._version import __version__
    from . import notation
    from . import tensors
    from . import utils
    from . import material
    from . import fabric_tensors
    from . import visualization
else:
    import notation
    import tensors
    import utils
    import material
    import fabric_tensors
    import visualization
    import _version
    __version__ = _version.__version__


