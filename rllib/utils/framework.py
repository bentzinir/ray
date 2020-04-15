import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Represents a generic tensor type.
TensorType = Any


def check_framework(framework="tf"):
    """
    Checks, whether the given framework is "valid", meaning, whether all
    necessary dependencies are installed. Errors otherwise.

    Args:
        framework (str): Once of "tf", "torch", or None.

    Returns:
        str: The input framework string.
    """
    if framework == "tf":
        if tf is None:
            raise ImportError("Could not import tensorflow.")
    elif framework == "torch":
        if torch is None:
            raise ImportError("Could not import torch.")
    else:
        assert framework is None
    return framework


def try_import_tf(error=False):
    """
    Args:
        error (bool): Whether to raise an error if tf cannot be imported.

    Returns:
        The tf module (either from tf2.0.compat.v1 OR as tf1.x.
    """
    # Make sure, these are reset after each test case
    # that uses them: del os.environ["RLLIB_TEST_NO_TF_IMPORT"]
    if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
        logger.warning("Not importing TensorFlow for test purposes")
        return None

    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Try to reuse already imported tf module. This will avoid going through
    # the initial import steps below and thereby switching off v2_behavior
    # (switching off v2 behavior twice breaks all-framework tests for eager).
    if "tensorflow" in sys.modules:
        tf_module = sys.modules["tensorflow"]
        # Try "reducing" tf to tf.compat.v1.
        try:
            tf_module = tf_module.compat.v1
        # No compat.v1 -> return tf as is.
        except AttributeError:
            pass
        return tf_module

    # Just in case. We should not go through the below twice.
    assert "tensorflow" not in sys.modules

    try:
        # Try "reducing" tf to tf.compat.v1.
        import tensorflow.compat.v1 as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        # Disable v2 eager mode.
        tf.disable_v2_behavior()
        return tf
    except ImportError:
        try:
            import tensorflow as tf
            return tf
        except ImportError as e:
            if error:
                raise e
            return None


def tf_function(tf_module):
    """Conditional decorator for @tf.function.

    Use @tf_function(tf) instead to avoid errors if tf is not installed."""

    # The actual decorator to use (pass in `tf` (which could be None)).
    def decorator(func):
        # If tf not installed -> return function as is (won't be used anyways).
        if tf_module is None or tf_module.executing_eagerly():
            return func
        # If tf installed, return @tf.function-decorated function.
        return tf_module.function(func)

    return decorator


def try_import_tfp(error=False):
    """
    Args:
        error (bool): Whether to raise an error if tfp cannot be imported.

    Returns:
        The tfp module.
    """
    if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
        logger.warning("Not importing TensorFlow Probability for test "
                       "purposes.")
        return None

    try:
        import tensorflow_probability as tfp
        return tfp
    except ImportError as e:
        if error:
            raise e
        return None


# Fake module for torch.nn.
class NNStub:
    def __init__(self, *a, **kw):
        # Fake nn.functional module within torch.nn.
        self.functional = None
        self.Module = ModuleStub


# Fake class for torch.nn.Module to allow it to be inherited from.
class ModuleStub:
    def __init__(self, *a, **kw):
        raise ImportError("Could not import `torch`.")


def try_import_torch(error=False):
    """
    Args:
        error (bool): Whether to raise an error if torch cannot be imported.

    Returns:
        tuple: torch AND torch.nn modules.
    """
    if "RLLIB_TEST_NO_TORCH_IMPORT" in os.environ:
        logger.warning("Not importing Torch for test purposes.")
        return _torch_stubs()

    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        if error:
            raise e
        return _torch_stubs()


def _torch_stubs():
    nn = NNStub()
    return None, nn


def get_variable(value,
                 framework="tf",
                 trainable=False,
                 tf_name="unnamed-variable",
                 torch_tensor=False,
                 device=None):
    """
    Args:
        value (any): The initial value to use. In the non-tf case, this will
            be returned as is.
        framework (str): One of "tf", "torch", or None.
        trainable (bool): Whether the generated variable should be
            trainable (tf)/require_grad (torch) or not (default: False).
        tf_name (str): For framework="tf": An optional name for the
            tf.Variable.
        torch_tensor (bool): For framework="torch": Whether to actually create
            a torch.tensor, or just a python value (default).

    Returns:
        any: A framework-specific variable (tf.Variable, torch.tensor, or
            python primitive).
    """
    if framework == "tf":
        import tensorflow as tf
        dtype = getattr(
            value, "dtype", tf.float32
            if isinstance(value, float) else tf.int32
            if isinstance(value, int) else None)
        return tf.compat.v1.get_variable(
            tf_name, initializer=value, dtype=dtype, trainable=trainable)
    elif framework == "torch" and torch_tensor is True:
        torch, _ = try_import_torch()
        var_ = torch.from_numpy(value).to(device)
        var_.requires_grad = trainable
        return var_
    # torch or None: Return python primitive.
    return value


def get_activation_fn(name, framework="tf"):
    """
    Returns a framework specific activation function, given a name string.

    Args:
        name (str): One of "relu" (default), "tanh", or "linear".
        framework (str): One of "tf" or "torch".

    Returns:
        A framework-specific activtion function. e.g. tf.nn.tanh or
            torch.nn.ReLU. Returns None for name="linear".
    """
    if framework == "torch":
        _, nn = try_import_torch()
        if name == "linear":
            return None
        elif name == "relu":
            return nn.ReLU
        elif name == "tanh":
            return nn.Tanh
    else:
        if name == "linear":
            return None
        tf = try_import_tf()
        fn = getattr(tf.nn, name, None)
        if fn is not None:
            return fn

    raise ValueError("Unknown activation ({}) for framework={}!".format(
        name, framework))


# This call should never happen inside a module's functions/classes
# as it would re-disable tf-eager.
tf = try_import_tf()
torch, _ = try_import_torch()
