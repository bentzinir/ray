from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import funcsigs
from funcsigs import Parameter
import logging

from ray.utils import is_cython

# Logger for this module. It should be configured at the entry point
# into the program using Ray. Ray provides a default configuration at
# entry/init points.
logger = logging.getLogger(__name__)

RayParameter = namedtuple(
    "RayParameter",
    ["name", "kind_int", "default", "annotation", "partial_kwarg"])
"""This class is used to represent a function parameter in Ray.

Note that this is different from the funcsigs.Parameter object because
we replace the funcsigs ParameterKind with an int. This is needed because
ParameterKind objects are currently non-serializable and the package is not
being updated. Replacement is done in `_scrub_parameters` and
`_restore_parameters`.

Attributes:
    name (str): The name of the parameter as a string.
    kind (int): Describes how argument values are bound to the parameter. See
        funcsigs.Parameter and `_convert_to_parameter_kind`.
    default (object): The default value for the parameter if specified. If the
        parameter has no default value, this attribute is not set.
    annotation: The annotation for the parameter if specified.  If the
        parameter has no annotation, this attribute is not set.
    partial_kwarg (bool): True if the parameter is mapped
        by 'functools.partial'.
"""

DUMMY_TYPE = "__RAY_DUMMY__"


def get_signature(func):
    """Get signature parameters.

    Support Cython functions by grabbing relevant attributes from the Cython
    function and attaching to a no-op function. This is somewhat brittle, since
    funcsigs may change, but given that funcsigs is written to a PEP, we hope
    it is relatively stable. Future versions of Python may allow overloading
    the inspect 'isfunction' and 'ismethod' functions / create ABC for Python
    functions. Until then, it appears that Cython won't do anything about
    compatability with the inspect module.

    Args:
        func: The function whose signature should be checked.

    Returns:
        A function signature object, which includes the names of the keyword
            arguments as well as their default values.

    Raises:
        TypeError: A type error if the signature is not supported
    """
    # The first condition for Cython functions, the latter for Cython instance
    # methods
    if is_cython(func):
        attrs = [
            "__code__", "__annotations__", "__defaults__", "__kwdefaults__"
        ]

        if all(hasattr(func, attr) for attr in attrs):
            original_func = func

            def func():
                return

            for attr in attrs:
                setattr(func, attr, getattr(original_func, attr))
        else:
            raise TypeError("{!r} is not a Python function we can process"
                            .format(func))

    return funcsigs.signature(func)


def extract_signature(func, ignore_first=False):
    """Extract the function signature from the function.

    Args:
        func: The function whose signature should be extracted.
        ignore_first: True if the first argument should be ignored. This should
            be used when func is a method of a class.

    Returns:
        List of RayParameter objects representing the function signature.
    """
    signature_parameters = list(get_signature(func).parameters.values())

    if ignore_first:
        if len(signature_parameters) == 0:
            raise Exception("Methods must take a 'self' argument, but the "
                            "method '{}' does not have one.".format(
                                func.__name__))
        signature_parameters = signature_parameters[1:]

    return _scrub_parameters(signature_parameters)


def flatten_args(signature_parameters, args, kwargs):
    """Validates the arguments against the signature and flattens them.

    The flat list representation is a serializable format for arguments.
    Since the flatbuffer representation of function arguments is a list, we
    combine both keyword arguments and positional arguments. We represent
    this with two entries per argument value - [DUMMY_TYPE, x] for positional
    arguments and [KEY, VALUE] for keyword arguments. See the below example.
    See `recover_args` for logic restoring the flat list back to args/kwargs.

    Args:
        signature_parameters (list): The list of RayParameter objects
            representing the function signature, obtained from
            `extract_signature`.
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.

    Returns:
        List of args and kwargs. Non-keyword arguments are prefixed
            by internal enum DUMMY_TYPE.

    Raises:
        TypeError: Raised if arguments do not fit in the function signature.

    Example:
        >>> flatten_args([1, 2, 3], {"a": 4})
        [None, 1, None, 2, None, 3, "a", 4]
    """
    restored = _restore_parameters(signature_parameters)
    reconstructed_signature = funcsigs.Signature(parameters=restored)
    try:
        reconstructed_signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(str(exc))
    list_args = []
    for arg in args:
        list_args += [DUMMY_TYPE, arg]

    for keyword, arg in kwargs.items():
        list_args += [keyword, arg]
    return list_args


def recover_args(flattened_args):
    """Recreates `args` and `kwargs` from the flattened arg list.

    Args:
        flattened_args: List of args and kwargs. This should be the output of
            `flatten_args`.

    Returns:
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.
    """
    assert len(flattened_args) % 2 == 0, (
        "Flattened arguments need to be even-numbered. See `flatten_args`.")
    args = []
    kwargs = {}
    for name_index in range(0, len(flattened_args), 2):
        name, arg = flattened_args[name_index], flattened_args[name_index + 1]
        if name == DUMMY_TYPE:
            args.append(arg)
        else:
            kwargs[name] = arg

    return args, kwargs


def _scrub_parameters(parameters):
    """Returns a scrubbed list of RayParameters."""
    return [
        RayParameter(
            name=param.name,
            kind_int=_convert_from_parameter_kind(param.kind),
            default=param.default,
            annotation=param.annotation,
            partial_kwarg=param._partial_kwarg) for param in parameters
    ]


def _restore_parameters(ray_parameters):
    """Reconstructs the funcsigs.Parameter objects."""
    return [
        Parameter(
            rayparam.name,
            _convert_to_parameter_kind(rayparam.kind_int),
            default=rayparam.default,
            annotation=rayparam.annotation,
            _partial_kwarg=rayparam.partial_kwarg)
        for rayparam in ray_parameters
    ]


def _convert_from_parameter_kind(kind):
    if kind == Parameter.POSITIONAL_ONLY:
        return 0
    if kind == Parameter.POSITIONAL_OR_KEYWORD:
        return 1
    if kind == Parameter.VAR_POSITIONAL:
        return 2
    if kind == Parameter.KEYWORD_ONLY:
        return 3
    if kind == Parameter.VAR_KEYWORD:
        return 4


def _convert_to_parameter_kind(value):
    if value == 0:
        return Parameter.POSITIONAL_ONLY
    if value == 1:
        return Parameter.POSITIONAL_OR_KEYWORD
    if value == 2:
        return Parameter.VAR_POSITIONAL
    if value == 3:
        return Parameter.KEYWORD_ONLY
    if value == 4:
        return Parameter.VAR_KEYWORD
