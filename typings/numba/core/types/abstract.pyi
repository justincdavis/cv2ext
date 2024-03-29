"""
This type stub file was generated by pyright.
"""

import weakref
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
from functools import cached_property

_typecodes = ...
_typecache: ptDict[weakref.ref, weakref.ref] = ...
class _TypeMetaclass(ABCMeta):
    """
    A metaclass that will intern instances after they are created.
    This is done by first creating a new instance (including calling
    __init__, which sets up the required attributes for equality
    and hashing), then looking it up in the _typecache registry.
    """
    def __init__(cls, name, bases, orig_vars) -> None:
        ...
    
    def __call__(cls, *args, **kwargs):
        """
        Instantiate *cls* (a Type subclass, presumably) and intern it.
        If an interned instance already exists, it is returned, otherwise
        the new instance is returned.
        """
        ...
    


class Type(metaclass=_TypeMetaclass):
    """
    The base class for all Numba types.
    It is essential that proper equality comparison is implemented.  The
    default implementation uses the "key" property (overridable in subclasses)
    for both comparison and hashing, to ensure sane behaviour.
    """
    mutable = ...
    reflected = ...
    def __init__(self, name) -> None:
        ...
    
    @property
    def key(self): # -> Any:
        """
        A property used for __eq__, __ne__ and __hash__.  Can be overridden
        in subclasses.
        """
        ...
    
    @property
    def mangling_args(self): # -> tuple[Any, tuple[()]]:
        """
        Returns `(basename, args)` where `basename` is the name of the type
        and `args` is a sequence of parameters of the type.

        Subclass should override to specialize the behavior.
        By default, this returns `(self.name, ())`.
        """
        ...
    
    def __repr__(self): # -> Any:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __ne__(self, other) -> bool:
        ...
    
    def __reduce__(self): # -> tuple[Callable[..., Any], tuple[str | Any, str | Any, str | Any]]:
        ...
    
    def unify(self, typingctx, other): # -> None:
        """
        Try to unify this type with the *other*.  A third type must
        be returned, or None if unification is not possible.
        Only override this if the coercion logic cannot be expressed
        as simple casting rules.
        """
        ...
    
    def can_convert_to(self, typingctx, other): # -> None:
        """
        Check whether this type can be converted to the *other*.
        If successful, must return a string describing the conversion, e.g.
        "exact", "promote", "unsafe", "safe"; otherwise None is returned.
        """
        ...
    
    def can_convert_from(self, typingctx, other): # -> None:
        """
        Similar to *can_convert_to*, but in reverse.  Only needed if
        the type provides conversion from other types.
        """
        ...
    
    def is_precise(self): # -> Literal[True]:
        """
        Whether this type is precise, i.e. can be part of a successful
        type inference.  Default implementation returns True.
        """
        ...
    
    def augment(self, other): # -> None:
        """
        Augment this type with the *other*.  Return the augmented type,
        or None if not supported.
        """
        ...
    
    def __call__(self, *args): # -> Signature:
        ...
    
    def __getitem__(self, args): # -> Array:
        """
        Return an array of this type.
        """
        ...
    
    def cast_python_value(self, args):
        ...
    
    @property
    def is_internal(self): # -> bool:
        """ Returns True if this class is an internally defined Numba type by
        virtue of the module in which it is instantiated, False else."""
        ...
    
    def dump(self, tab=...): # -> None:
        ...
    


class Dummy(Type):
    """
    Base class for types that do not really have a representation and are
    compatible with a void*.
    """
    ...


class Hashable(Type):
    """
    Base class for hashable types.
    """
    ...


class Number(Hashable):
    """
    Base class for number types.
    """
    def unify(self, typingctx, other): # -> Record | UnicodeCharSeq | CharSeq | NPTimedelta | NPDatetime | NestedArray | None:
        """
        Unify the two number types using Numpy's rules.
        """
        ...
    


class Callable(Type):
    """
    Base class for callables.
    """
    @abstractmethod
    def get_call_type(self, context, args, kws): # -> None:
        """
        Using the typing *context*, resolve the callable's signature for
        the given arguments.  A signature object is returned, or None.
        """
        ...
    
    @abstractmethod
    def get_call_signatures(self): # -> None:
        """
        Returns a tuple of (list of signatures, parameterized)
        """
        ...
    
    @abstractmethod
    def get_impl_key(self, sig): # -> None:
        """
        Returns the impl key for the given signature
        """
        ...
    


class DTypeSpec(Type):
    """
    Base class for types usable as "dtype" arguments to various Numpy APIs
    (e.g. np.empty()).
    """
    @abstractproperty
    def dtype(self): # -> None:
        """
        The actual dtype denoted by this dtype spec (a Type instance).
        """
        ...
    


class IterableType(Type):
    """
    Base class for iterable types.
    """
    @abstractproperty
    def iterator_type(self): # -> None:
        """
        The iterator type obtained when calling iter() (explicitly or implicitly).
        """
        ...
    


class Sized(Type):
    """
    Base class for objects that support len()
    """
    ...


class ConstSized(Sized):
    """
    For types that have a constant size
    """
    @abstractmethod
    def __len__(self): # -> None:
        ...
    


class IteratorType(IterableType):
    """
    Base class for all iterator types.
    Derived classes should implement the *yield_type* attribute.
    """
    def __init__(self, name, **kwargs) -> None:
        ...
    
    @abstractproperty
    def yield_type(self): # -> None:
        """
        The type of values yielded by the iterator.
        """
        ...
    
    @property
    def iterator_type(self): # -> Self:
        ...
    


class Container(Sized, IterableType):
    """
    Base class for container types.
    """
    ...


class Sequence(Container):
    """
    Base class for 1d sequence types.  Instances should have the *dtype*
    attribute.
    """
    ...


class MutableSequence(Sequence):
    """
    Base class for 1d mutable sequence types.  Instances should have the
    *dtype* attribute.
    """
    ...


class ArrayCompatible(Type):
    """
    Type class for Numpy array-compatible objects (typically, objects
    exposing an __array__ method).
    Derived classes should implement the *as_array* attribute.
    """
    array_priority = ...
    @abstractproperty
    def as_array(self): # -> None:
        """
        The equivalent array type, for operations supporting array-compatible
        objects (such as ufuncs).
        """
        ...
    
    @cached_property
    def ndim(self):
        ...
    
    @cached_property
    def layout(self):
        ...
    
    @cached_property
    def dtype(self):
        ...
    


class Literal(Type):
    """Base class for Literal types.
    Literal types contain the original Python value in the type.

    A literal type should always be constructed from the `literal(val)`
    function.
    """
    ctor_map: ptDict[type, ptType[Literal]] = ...
    _literal_type_cache = ...
    def __init__(self, value) -> None:
        ...
    
    @property
    def literal_value(self):
        ...
    
    @property
    def literal_type(self): # -> Buffer | Any | ExternalFunctionPointer | Opaque | ExternalFunction | Module:
        ...
    


class TypeRef(Dummy):
    """Reference to a type.

    Used when a type is passed as a value.
    """
    def __init__(self, instance_type) -> None:
        ...
    
    @property
    def key(self): # -> Any:
        ...
    


class InitialValue:
    """
    Used as a mixin for a type will potentially have an initial value that will
    be carried in the .initial_value attribute.
    """
    def __init__(self, initial_value) -> None:
        ...
    
    @property
    def initial_value(self): # -> Any:
        ...
    


class Poison(Type):
    """
    This is the "bottom" type in the type system. It won't unify and it's
    unliteral version is Poison of itself. It's advisable for debugging purposes
    to call the constructor with the type that's being poisoned (for whatever
    reason) but this isn't strictly required.
    """
    def __init__(self, ty) -> None:
        ...
    
    def __unliteral__(self): # -> Poison:
        ...
    
    def unify(self, typingctx, other): # -> None:
        ...
    


