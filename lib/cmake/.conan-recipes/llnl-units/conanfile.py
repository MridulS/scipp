
import os

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps
from conan.tools.files import copy, rm, rmdir, get

class UnitsConan(ConanFile):
    name = "llnl-units"
    version = "0.9.1"
    license = "BSD-3"
    url = "https://github.com/llnl/units"
    homepage = "https://units.readthedocs.io"
    description = (
        "A run-time C++ library for working with units "
        "of measurement and conversions between them "
        "and with string representations of units "
        "and measurements"
    )
    topics = (
        "units",
        "dimensions",
        "quantities",
        "physical-units",
        "dimensional-analysis",
        "run-time",
    )
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "base_type": ["uint32_t", "uint64_t"],
        "namespace": ["ANY"],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "base_type": "uint32_t",
        "namespace": None,
    }
    @property
    def _min_cppstd(self):
        return 17

    @property
    def _compilers_minimum_version(self):
        return {
            "apple-clang": "10",
            "clang": "7",
            "gcc": "7",
            "msvc": "191",
            "Visual Studio": "15",
        }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def source(self):
        get(self,
            url="https://github.com/LLNL/units/archive/refs/tags/v0.9.1.tar.gz",
            sha256= "7edb83613a07cf55873f22d61c0062e46db6f8cb27d137866858811ec2e1ad4f", strip_root=True)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.preprocessor_definitions["UNITS_CMAKE_PROJECT_NAME"] = "LLNL-UNITS"
        tc.preprocessor_definitions["UNITS_ENABLE_TESTS"] = "OFF"
        tc.preprocessor_definitions["UNITS_BUILD_SHARED_LIBRARY"] = self.options.shared
        tc.preprocessor_definitions[
            "UNITS_BUILD_STATIC_LIBRARY"
        ] = not self.options.shared
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE",
            self.source_folder,
            os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        rmdir(self, os.path.join(self.package_folder, "share"))
        rm(self, "*.la", os.path.join(self.package_folder, "lib"))
        rm(self, "*.pdb", os.path.join(self.package_folder, "lib"))
        rm(self, "*.pdb", os.path.join(self.package_folder, "bin"))

    def package_info(self):
        self.cpp_info.libs = ["units"]
        units_namespace = self.options.get_safe("namespace")
        self.cpp_info.defines = [f"UNITS_BASE_TYPE={self.options.base_type}"]
        if units_namespace:
            self.cpp_info.defines.append(f"UNITS_NAMESPACE={units_namespace}")
        self.cpp_info.set_property("cmake_file_name", "llnl-units")
        self.cpp_info.set_property("cmake_target_name", "llnl-units::llnl-units")