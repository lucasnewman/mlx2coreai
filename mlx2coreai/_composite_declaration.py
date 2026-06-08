# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

from coreai._compiler.ir import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    Context,
    DictAttr,
    F32Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    Location,
    NamedAttribute,
    StringAttr,
)


@dataclass
class CompositeDeclaration:
    """
    Composite op declaration Attribute.

    Holds information about a composite op and provides a way to construct the
    Core AI Attribute and parse the attribute to extract all data.

    Parameters
    ----------
    name: str
        The name of the composite op.
    attributes: dict
        All attributes for the composite ops.
    """

    name: str
    attributes: dict[str, Any]

    def _get_array_attr(self, vals: list[Any], context: Context) -> ArrayAttr:
        """Return ArrayAttr for values in list."""
        results: list[Any] = []
        for v in vals:
            if isinstance(v, bool):
                results.append(BoolAttr.get(v, context))
            elif isinstance(v, int):
                results.append(IntegerAttr.get(IntegerType.get_signed(64, context), v))
            elif isinstance(v, float):
                results.append(FloatAttr.get(F32Type.get(context), v))
            elif isinstance(v, dict):
                results.append(self._dict_to_dict_attr(v, context))
            elif isinstance(v, str):
                results.append(StringAttr.get(str(v), context))
            elif isinstance(v, list):
                results.append(self._get_array_attr(v, context))
        return ArrayAttr.get(results, context)

    @classmethod
    def _array_attr_to_list(cls, array_attr: ArrayAttr) -> list[Any]:
        """Return list corresponding to passed ArrayAttr."""
        result: list[Any] = []
        for attr in array_attr:
            if isinstance(attr, IntegerAttr):
                result.append(IntegerAttr(attr).value)
            elif isinstance(attr, FloatAttr):
                result.append(FloatAttr(attr).value)
            elif isinstance(attr, BoolAttr):
                result.append(BoolAttr(attr).value)
            elif isinstance(attr, StringAttr):
                result.append(StringAttr(attr).value)
            else:
                msg = "Unsupported value provided in composite declaration."
                raise TypeError(msg)
        return result

    def _dict_to_dict_attr(self, py_dict: dict[str, Any], context: Context) -> DictAttr:
        """Return input python dict as DictAttr."""
        updated_dict: dict[str, Any] = {}
        for k, v in py_dict.items():
            if isinstance(v, bool):
                updated_dict[k] = BoolAttr.get(v, context)
            elif isinstance(v, int):
                updated_dict[k] = IntegerAttr.get(
                    IntegerType.get_signed(64, context), v
                )
            elif isinstance(v, float):
                updated_dict[k] = FloatAttr.get(F32Type.get(context), v)
            elif isinstance(v, dict):
                updated_dict[k] = self._dict_to_dict_attr(v, context)
            elif isinstance(v, str):
                updated_dict[k] = StringAttr.get(str(v), context)
            elif isinstance(v, list):
                updated_dict[k] = self._get_array_attr(v, context)
            else:
                msg = f"Unsupported value provided in composite declaration {v}."
                raise TypeError(msg)
        return DictAttr.get(updated_dict, context)

    @classmethod
    def _dict_attr_to_dict(cls, dict_attr: DictAttr) -> dict[str, Any]:
        py_dict: dict[str, Any] = {}
        for named_attr in dict_attr:
            named_attr = cast("NamedAttribute", named_attr)
            if isinstance(named_attr.attr, IntegerAttr):
                py_dict[named_attr.name] = IntegerAttr(named_attr.attr).value
            elif isinstance(named_attr.attr, FloatAttr):
                py_dict[named_attr.name] = FloatAttr(named_attr.attr).value
            elif isinstance(named_attr.attr, BoolAttr):
                py_dict[named_attr.name] = BoolAttr(named_attr.attr).value
            elif isinstance(named_attr.attr, StringAttr):
                py_dict[named_attr.name] = StringAttr(named_attr.attr).value
            elif isinstance(named_attr.attr, ArrayAttr):
                py_dict[named_attr.name] = CompositeDeclaration._array_attr_to_list(
                    ArrayAttr(named_attr.attr)
                )
            elif isinstance(named_attr.attr, DictAttr):
                py_dict[named_attr.name] = CompositeDeclaration._dict_attr_to_dict(
                    named_attr.attr
                )
            else:
                msg = f"Unsupported attribute provided in composite declaration {named_attr.attr}."
                raise TypeError(msg)
        return py_dict

    def to_coreai_attr(self, context: Context) -> Attribute:
        """Return Core AI Attribute representation."""
        with Location.unknown(context):
            attrs = self._dict_to_dict_attr(self.attributes, context)
        return Attribute.parse(
            f'#coreai.composite_declaration<"{self.name}" = {attrs!s}>',
            context=context,
        )

    @classmethod
    def from_coreai_attr(cls, value: str, context: Context) -> CompositeDeclaration:
        """Parse composite op attribute from string and return CompositeDeclaration."""
        min_number_of_splits = 2
        name_splits = value.split('"')
        if len(name_splits) < min_number_of_splits:
            return cls("", {})
        name = name_splits[1].strip()
        dict_start = value.find("{")
        if dict_start == -1:
            return cls("", {})

        attrs = DictAttr.parse(
            value.strip()[dict_start : len(value) - 1],
            context=context,
        )
        return cls(
            name,
            CompositeDeclaration._dict_attr_to_dict(cast("DictAttr", attrs)),
        )


def generate_composite_decl(
    context: Context,
    composite_name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    op_attributes: dict,
    version=1,
):
    """
    Helper for generating a CompositeDeclaration attribute.

    Args:
        context: A Core AI context required to construct the `Attribute`
        composite_name: The name of the composite operation
        input_names: The names of the composite arguments
        output_names: The names of the composite results
        op_attributes: A dictionary containing any attributes required by the composite
                       i.e: The SDPA composite has an 'is_causal' attribute.

    """
    assert isinstance(input_names, Sequence), (
        "input_names must be a Sequence of strings"
    )
    assert isinstance(output_names, Sequence), (
        "output_names must be a Sequence of strings"
    )

    for name in input_names + output_names:
        assert isinstance(name, str), (
            "input_names/output_names must solely contain strings"
        )

    op_attributes["version"] = version
    return CompositeDeclaration(
        composite_name,
        {
            "input_names": input_names,
            "output_names": output_names,
            "op_attrs": op_attributes,
        },
    ).to_coreai_attr(context)
