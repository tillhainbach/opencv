#!/usr/bin/env python


from __future__ import print_function
import os, sys, re, string, io
from typing import List, Union, TypedDict, Set, Generator
from hdr_parser import CppHeaderParser
# the list only for debugging. The real list, used in the real OpenCV build, is specified in CMakeLists.txt
opencv_hdr_list = [
"../../core/include/opencv2/core.hpp",
"../../core/include/opencv2/core/mat.hpp",
"../../core/include/opencv2/core/ocl.hpp",
"../../flann/include/opencv2/flann/miniflann.hpp",
"../../ml/include/opencv2/ml.hpp",
"../../imgproc/include/opencv2/imgproc.hpp",
"../../calib3d/include/opencv2/calib3d.hpp",
"../../features2d/include/opencv2/features2d.hpp",
"../../video/include/opencv2/video/tracking.hpp",
"../../video/include/opencv2/video/background_segm.hpp",
"../../objdetect/include/opencv2/objdetect.hpp",
"../../imgcodecs/include/opencv2/imgcodecs.hpp",
"../../videoio/include/opencv2/videoio.hpp",
"../../highgui/include/opencv2/highgui.hpp",
]

"""Parse the declaration list into descriptive maps.

Each declaration is:
    [funcname, return_value_type /* in C, not in Python */,
     <list_of_modifiers>,
     <list_of_arguments>,
     original_return_type,
     docstring],

where each element of <list_of_arguments> is 4-element list itself:
    [argtype, argname, default_value /* or "" if none */, <list_of_modifiers>],

where the list of modifiers is yet another nested list of strings:
    currently recognized are:
        "/O" for output argument
        "/S" for static (i.e. class) methods
        and "/A value" for the plain C arrays with counters

original_return_type is None if the original_return_type is the same as
return_value_type
"""


class ArgumentDict(TypedDict):
    name: str
    type: str
    default_value: str
    modifiers: List[str]

class ClassDict(TypedDict):
    name: str
    type: str
    base_class: str
    class_properties: List[ArgumentDict]
    instance_properties: List[ArgumentDict]
    methods: List[FunctionDict]

class FunctionDict(TypedDict):
    name: str
    type: str
    arguments: List[ArgumentDict]
    return_type: str
    modifiers: List[ArgumentDict]

class EnumDict(TypedDict):
    name: str
    type: str
    properties: List[ArgumentDict]

class DeclarationDict(TypedDict):
    name: str
    c_return_value_Type: str
    modifiers: List[str]
    arguments: List[ArgumentDict]
    original_return_type: str
    docstring: str


def _arguments_list_to_dict(argument_list) -> ArgumentDict:

    argument: ArgumentDict = {
        "type": argument_list[0],
        "name": argument_list[1],
        "default_value": argument_list[2],
        "modifiers": argument_list[3]
    }

    return argument

def _declaration_list_to_dict(declaration_list) -> DeclarationDict:

    arguments: List[ArgumentDict] = [
        _arguments_list_to_dict(argument_list)
            for argument_list in declaration_list[3]
    ]

    declaration: DeclarationDict = {
        "funcname": declaration_list[0],
        "c_return_value_type": declaration_list[1],
        "modifiers": declaration_list[2],
        "arguments": arguments,
        "original_type": declaration_list[4],
        "docstrin": declaration_list[5]
    }

    return declaration


class Argument:

    def __init__(self, argument_list) -> None:
        self.type: str = argument_list[0]
        self.name: str = argument_list[1]
        self.default_value: str = argument_list[2]
        self.modifiers: List[str] = argument_list[3]

    def _convert_to_none(self, value: str) -> str:
        if value == "":
            return "None"
        return value

    def to_python_typed_argument(self) -> str:
        """Return a python typed arguemnt."""
        typed_argument = "{name}: {type}".format(
            name=self.name,
            type=self.type,
        )
        if self.default_value:
            typed_argument += " = {default_value}".format(
                default_value=self.default_value
            )

        return typed_argument


class Declaration:

    def __init__(self, declaration_list: List[str],
                 namespaces: Set[str]) -> None:
        self.name: str = declaration_list[0].replace("cv.", "")
        self.basename: str = ""
        self.object_type: str = ""
        self._parse_name(namespaces)
        self.c_return_value_type: str = declaration_list[1]
        self.modifiers: List[str] = declaration_list[2]
        self.arguments: List[Argument] = [
            Argument(argument_list) for argument_list in declaration_list[3]
        ]
        self.original_return_type: str = declaration_list[4]
        self.docstring: str = declaration_list[5]

    def to_python_stub(self) -> str:
        """Return a python stub."""
        args = ",\n\t".join(
            arg.to_python_typed_argument() for arg in self.arguments
        )
        if len(args) > 10:
            args = "\n\t" + args + "\n"
        python_stub = "def {name}({args}) -> {return_type}: ...".format(
            name=self.name,
            args=args,
            return_type=self.original_return_type
        )

        return python_stub

    def _parse_name(self, namespaces: Set[str]) -> None:
        """Return the object type as string."""

        object_type: List[str] = self.name.split(" ")
        self.name = object_type[-1]
        self.object_type = "function" if len(object_type) == 1 else object_type[0]

        for namespace in namespaces:
            if self.name.startswith(namespace):
                self.name = self.name.replace(".", "_", 1)

        if "." in self.name:
            try:
                self.name, self.basename = self.name.split(".")
            except:
                print(self.name, namespaces)
                raise

        if self.name == self.basename:
            self.basename = "__init__"


def print_decls_as_stubs(decls: list) -> None:
    """Print stubs for function"""

    for d in decls:
        declaration = Declaration(d)
        print(declaration.to_python_stub())


class DeclarationParser:
    """Parser for parsing declaration list generator by the CppHeaderParser."""

    def __init__(self) -> None:
        self.namespaces: Set[str] = set()

    def parse(self, files) -> None:
        """Parse files."""

        parser = CppHeaderParser(generate_umat_decls=True,
                                 generate_gpumat_decls=True)

        for file in files:
            decls = parser.parse(file)
            namespaces = [namespace.replace("cv.", "")
                for namespace in parser.namespaces if namespace != "cv"]

            if len(decls) == 0:
                continue

            decl_parser = declaration_parser()
            next(decl_parser)
            for decl in decls:
                declaration = Declaration(decl, namespaces)
                decl_parser.send(declaration)


def declaration_parser() -> Generator[None, Declaration, None]:
    """Parse declaration."""

    while True:
        declaration = yield
        name = declaration.name
        basename = declaration.basename
        if declaration.object_type == "function" and basename is None:
            print("def {}(): ...".format(name))
        if declaration.object_type == "class" or basename:
            class_name = name
            print("class {}:".format(name))
            if basename:
                print("\tdef {}(self): ...".format(basename))
            while True:
                declaration = yield
                name = declaration.name
                basename = declaration.basename
                if declaration.object_type != "function" or not basename:
                    break
                print("\tdef {}(self): ...".format(basename))
        if declaration.object_type == "enum":
            print("class {}(IntFlag):".format(declaration.name))
            for arg in declaration.arguments:
                name = arg.type.replace("const cv.", "")
                print("\t{}: int = {}".format(name, arg.name))


def parser_sink() -> Generator[None, Tuple(Declaration, type), Dict]:
    """Create dictionary of functions."""


    return dict


if __name__ == '__main__':
    parser = DeclarationParser()
    parser.parse(opencv_hdr_list)
    # print(len(decls))
    # print("namespaces:", " ".join(sorted(parser.namespaces)))
