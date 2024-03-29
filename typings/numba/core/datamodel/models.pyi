"""
This type stub file was generated by pyright.
"""

from numba.core.datamodel.registry import register_default
from numba.core import types

class DataModel:
    """
    DataModel describe how a FE type is represented in the LLVM IR at
    different contexts.

    Contexts are:

    - value: representation inside function body.  Maybe stored in stack.
    The representation here are flexible.

    - data: representation used when storing into containers (e.g. arrays).

    - argument: representation used for function argument.  All composite
    types are unflattened into multiple primitive types.

    - return: representation used for return argument.

    Throughput the compiler pipeline, a LLVM value is usually passed around
    in the "value" representation.  All "as_" prefix function converts from
    "value" representation.  All "from_" prefix function converts to the
    "value"  representation.

    """
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    @property
    def fe_type(self): # -> Any:
        ...
    
    def get_value_type(self):
        ...
    
    def get_data_type(self):
        ...
    
    def get_argument_type(self):
        """Return a LLVM type or nested tuple of LLVM type
        """
        ...
    
    def get_return_type(self):
        ...
    
    def as_data(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        """
        Takes one LLVM value
        Return a LLVM value or nested tuple of LLVM value
        """
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        """
        Takes a LLVM value or nested tuple of LLVM value
        Returns one LLVM value
        """
        ...
    
    def from_return(self, builder, value):
        ...
    
    def load_from_data_pointer(self, builder, ptr, align=...):
        """
        Load value from a pointer to data.
        This is the default implementation, sufficient for most purposes.
        """
        ...
    
    def traverse(self, builder): # -> list[Any]:
        """
        Traverse contained members.
        Returns a iterable of contained (types, getters).
        Each getter is a one-argument function accepting a LLVM value.
        """
        ...
    
    def traverse_models(self): # -> list[Any]:
        """
        Recursively list all models involved in this model.
        """
        ...
    
    def traverse_types(self): # -> list[Any]:
        """
        Recursively list all frontend types involved in this model.
        """
        ...
    
    def inner_models(self): # -> list[Any]:
        """
        List all *inner* models.
        """
        ...
    
    def get_nrt_meminfo(self, builder, value): # -> None:
        """
        Returns the MemInfo object or None if it is not tracked.
        It is only defined for types.meminfo_pointer
        """
        ...
    
    def has_nrt_meminfo(self): # -> Literal[False]:
        ...
    
    def contains_nrt_meminfo(self): # -> bool:
        """
        Recursively check all contained types for need for NRT meminfo.
        """
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __ne__(self, other) -> bool:
        ...
    


@register_default(types.Omitted)
class OmittedArgDataModel(DataModel):
    """
    A data model for omitted arguments.  Only the "argument" representation
    is defined, other representations raise a NotImplementedError.
    """
    def get_value_type(self): # -> LiteralStructType:
        ...
    
    def get_argument_type(self): # -> tuple[()]:
        ...
    
    def as_argument(self, builder, val): # -> tuple[()]:
        ...
    
    def from_argument(self, builder, val): # -> None:
        ...
    


@register_default(types.Boolean)
@register_default(types.BooleanLiteral)
class BooleanModel(DataModel):
    _bit_type = ...
    _byte_type = ...
    def get_value_type(self): # -> IntType:
        ...
    
    def get_data_type(self): # -> IntType:
        ...
    
    def get_return_type(self): # -> IntType:
        ...
    
    def get_argument_type(self): # -> IntType:
        ...
    
    def as_data(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    


class PrimitiveModel(DataModel):
    """A primitive type can be represented natively in the target in all
    usage contexts.
    """
    def __init__(self, dmm, fe_type, be_type) -> None:
        ...
    
    def get_value_type(self): # -> Any:
        ...
    
    def as_data(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    


class ProxyModel(DataModel):
    """
    Helper class for models which delegate to another model.
    """
    def get_value_type(self):
        ...
    
    def get_data_type(self):
        ...
    
    def get_return_type(self):
        ...
    
    def get_argument_type(self):
        ...
    
    def as_data(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    


@register_default(types.EnumMember)
@register_default(types.IntEnumMember)
class EnumModel(ProxyModel):
    """
    Enum members are represented exactly like their values.
    """
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.Opaque)
@register_default(types.PyObject)
@register_default(types.RawPointer)
@register_default(types.NoneType)
@register_default(types.StringLiteral)
@register_default(types.EllipsisType)
@register_default(types.Function)
@register_default(types.Type)
@register_default(types.Object)
@register_default(types.Module)
@register_default(types.Phantom)
@register_default(types.UndefVar)
@register_default(types.ContextManager)
@register_default(types.Dispatcher)
@register_default(types.ObjModeDispatcher)
@register_default(types.ExceptionClass)
@register_default(types.Dummy)
@register_default(types.ExceptionInstance)
@register_default(types.ExternalFunction)
@register_default(types.EnumClass)
@register_default(types.IntEnumClass)
@register_default(types.NumberClass)
@register_default(types.TypeRef)
@register_default(types.NamedTupleClass)
@register_default(types.DType)
@register_default(types.RecursiveCall)
@register_default(types.MakeFunctionLiteral)
@register_default(types.Poison)
class OpaqueModel(PrimitiveModel):
    """
    Passed as opaque pointers
    """
    _ptr_type = ...
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.MemInfoPointer)
class MemInfoModel(OpaqueModel):
    def inner_models(self): # -> list[Any]:
        ...
    
    def has_nrt_meminfo(self): # -> Literal[True]:
        ...
    
    def get_nrt_meminfo(self, builder, value):
        ...
    


@register_default(types.Integer)
@register_default(types.IntegerLiteral)
class IntegerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.Float)
class FloatModel(PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.CPointer)
class PointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.EphemeralPointer)
class EphemeralPointerModel(PointerModel):
    def get_data_type(self):
        ...
    
    def as_data(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def load_from_data_pointer(self, builder, ptr, align=...):
        ...
    


@register_default(types.EphemeralArray)
class EphemeralArrayModel(PointerModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_data_type(self): # -> ArrayType:
        ...
    
    def as_data(self, builder, value): # -> Constant:
        ...
    
    def from_data(self, builder, value):
        ...
    
    def load_from_data_pointer(self, builder, ptr, align=...):
        ...
    


@register_default(types.ExternalFunctionPointer)
class ExternalFuncPointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.UniTuple)
@register_default(types.NamedUniTuple)
@register_default(types.StarArgUniTuple)
class UniTupleModel(DataModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_value_type(self): # -> ArrayType:
        ...
    
    def get_data_type(self): # -> ArrayType:
        ...
    
    def get_return_type(self): # -> ArrayType:
        ...
    
    def get_argument_type(self): # -> tuple[Any, ...]:
        ...
    
    def as_argument(self, builder, value): # -> list[Any]:
        ...
    
    def from_argument(self, builder, value): # -> Constant:
        ...
    
    def as_data(self, builder, value): # -> Constant:
        ...
    
    def from_data(self, builder, value): # -> Constant:
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def traverse(self, builder): # -> list[tuple[Any, partial[Any]]]:
        ...
    
    def inner_models(self): # -> list[Any]:
        ...
    


class CompositeModel(DataModel):
    """Any model that is composed of multiple other models should subclass from
    this.
    """
    ...


class StructModel(CompositeModel):
    _value_type = ...
    _data_type = ...
    def __init__(self, dmm, fe_type, members) -> None:
        ...
    
    def get_member_fe_type(self, name): # -> Any:
        """
        StructModel-specific: get the Numba type of the field named *name*.
        """
        ...
    
    def get_value_type(self): # -> LiteralStructType:
        ...
    
    def get_data_type(self): # -> LiteralStructType:
        ...
    
    def get_argument_type(self): # -> tuple[Any, ...]:
        ...
    
    def get_return_type(self): # -> LiteralStructType:
        ...
    
    def as_data(self, builder, value): # -> Constant:
        """
        Converts the LLVM struct in `value` into a representation suited for
        storing into arrays.

        Note
        ----
        Current implementation rarely changes how types are represented for
        "value" and "data".  This is usually a pointless rebuild of the
        immutable LLVM struct value.  Luckily, LLVM optimization removes all
        redundancy.

        Sample usecase: Structures nested with pointers to other structures
        that can be serialized into  a flat representation when storing into
        array.
        """
        ...
    
    def from_data(self, builder, value): # -> Constant:
        """
        Convert from "data" representation back into "value" representation.
        Usually invoked when loading from array.

        See notes in `as_data()`
        """
        ...
    
    def load_from_data_pointer(self, builder, ptr, align=...): # -> Constant:
        ...
    
    def as_argument(self, builder, value): # -> tuple[Any, ...]:
        ...
    
    def from_argument(self, builder, value): # -> Constant:
        ...
    
    def as_return(self, builder, value): # -> Constant:
        ...
    
    def from_return(self, builder, value): # -> Constant:
        ...
    
    def get(self, builder, val, pos):
        """Get a field at the given position or the fieldname

        Args
        ----
        builder:
            LLVM IRBuilder
        val:
            value to be inserted
        pos: int or str
            field index or field name

        Returns
        -------
        Extracted value
        """
        ...
    
    def set(self, builder, stval, val, pos):
        """Set a field at the given position or the fieldname

        Args
        ----
        builder:
            LLVM IRBuilder
        stval:
            LLVM struct value
        val:
            value to be inserted
        pos: int or str
            field index or field name

        Returns
        -------
        A new LLVM struct with the value inserted
        """
        ...
    
    def get_field_position(self, field): # -> int | Any:
        ...
    
    @property
    def field_count(self): # -> int:
        ...
    
    def get_type(self, pos): # -> Any:
        """Get the frontend type (numba type) of a field given the position
         or the fieldname

        Args
        ----
        pos: int or str
            field index or field name
        """
        ...
    
    def get_model(self, pos):
        """
        Get the datamodel of a field given the position or the fieldname.

        Args
        ----
        pos: int or str
            field index or field name
        """
        ...
    
    def traverse(self, builder): # -> list[tuple[Any, partial[Any]]]:
        ...
    
    def inner_models(self): # -> tuple[Any, ...]:
        ...
    


@register_default(types.Complex)
class ComplexModel(StructModel):
    _element_type = ...
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.LiteralList)
@register_default(types.LiteralStrKeyDict)
@register_default(types.Tuple)
@register_default(types.NamedTuple)
@register_default(types.StarArgTuple)
class TupleModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.UnionType)
class UnionModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.Pair)
class PairModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.ListPayload)
class ListPayloadModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.List)
class ListModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.ListIter)
class ListIterModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.SetEntry)
class SetEntryModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.SetPayload)
class SetPayloadModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.Set)
class SetModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.SetIter)
class SetIterModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.Array)
@register_default(types.Buffer)
@register_default(types.ByteArray)
@register_default(types.Bytes)
@register_default(types.MemoryView)
@register_default(types.PyArray)
class ArrayModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.ArrayFlags)
class ArrayFlagsModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.NestedArray)
class NestedArrayModel(ArrayModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def as_storage_type(self): # -> ArrayType:
        """Return the LLVM type representation for the storage of
        the nestedarray.
        """
        ...
    


@register_default(types.Optional)
class OptionalModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_return_type(self):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def traverse(self, builder): # -> list[tuple[Any, Callable[..., Any]]]:
        ...
    


@register_default(types.Record)
class RecordModel(CompositeModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_value_type(self): # -> PointerType:
        """Passed around as reference to underlying data
        """
        ...
    
    def get_argument_type(self): # -> PointerType:
        ...
    
    def get_return_type(self): # -> PointerType:
        ...
    
    def get_data_type(self): # -> ArrayType:
        ...
    
    def as_data(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def load_from_data_pointer(self, builder, ptr, align=...):
        ...
    


@register_default(types.UnicodeCharSeq)
class UnicodeCharSeq(DataModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_value_type(self): # -> ArrayType:
        ...
    
    def get_data_type(self): # -> ArrayType:
        ...
    
    def as_data(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    


@register_default(types.CharSeq)
class CharSeq(DataModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_value_type(self): # -> ArrayType:
        ...
    
    def get_data_type(self): # -> ArrayType:
        ...
    
    def as_data(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    


class CContiguousFlatIter(StructModel):
    def __init__(self, dmm, fe_type, need_indices) -> None:
        ...
    


class FlatIter(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.UniTupleIter)
class UniTupleIter(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.misc.SliceLiteral)
@register_default(types.SliceType)
class SliceModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.NPDatetime)
@register_default(types.NPTimedelta)
class NPDatetimeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.ArrayIterator)
class ArrayIterator(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.EnumerateType)
class EnumerateType(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.ZipType)
class ZipType(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.RangeIteratorType)
class RangeIteratorType(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.Generator)
class GeneratorModel(CompositeModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_value_type(self): # -> PointerType:
        """
        The generator closure is passed around as a reference.
        """
        ...
    
    def get_argument_type(self): # -> PointerType:
        ...
    
    def get_return_type(self): # -> LiteralStructType:
        ...
    
    def get_data_type(self): # -> LiteralStructType:
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def as_data(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    


@register_default(types.ArrayCTypes)
class ArrayCTypesModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.RangeType)
class RangeModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.NumpyNdIndexType)
class NdIndexModel(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.NumpyFlatType)
def handle_numpy_flat_type(dmm, ty): # -> CContiguousFlatIter | FlatIter:
    ...

@register_default(types.NumpyNdEnumerateType)
def handle_numpy_ndenumerate_type(dmm, ty): # -> CContiguousFlatIter | FlatIter:
    ...

@register_default(types.BoundFunction)
def handle_bound_function(dmm, ty):
    ...

@register_default(types.NumpyNdIterType)
class NdIter(StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_default(types.DeferredType)
class DeferredStructModel(CompositeModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    
    def get_value_type(self): # -> IdentifiedStructType:
        ...
    
    def get_data_type(self): # -> IdentifiedStructType:
        ...
    
    def get_argument_type(self):
        ...
    
    def as_argument(self, builder, value):
        ...
    
    def from_argument(self, builder, value):
        ...
    
    def from_data(self, builder, value):
        ...
    
    def as_data(self, builder, value):
        ...
    
    def from_return(self, builder, value):
        ...
    
    def as_return(self, builder, value):
        ...
    
    def get(self, builder, value):
        ...
    
    def set(self, builder, value, content):
        ...
    
    def make_uninitialized(self, kind=...): # -> Constant:
        ...
    
    def traverse(self, builder): # -> list[tuple[Any, Callable[..., Any]]]:
        ...
    


@register_default(types.StructRefPayload)
class StructPayloadModel(StructModel):
    """Model for the payload of a mutable struct
    """
    def __init__(self, dmm, fe_typ) -> None:
        ...
    


class StructRefModel(StructModel):
    """Model for a mutable struct.
    A reference to the payload
    """
    def __init__(self, dmm, fe_typ) -> None:
        ...
    


