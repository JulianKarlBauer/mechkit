import sys

if sys.version_info < (3, 0):
    import notation
    import tensors
    import utils
    import material
    import fabric_tensors
    import visualization
    import operators
else:
    from . import notation
    from . import tensors
    from . import utils
    from . import material
    from . import fabric_tensors
    from . import visualization
    from . import operators

# import pkg_resources
#
# __version__ = pkg_resources.get_distribution("mechkit").version
