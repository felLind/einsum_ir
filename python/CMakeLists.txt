cmake_minimum_required(VERSION 3.20)
project(etops LANGUAGES CXX)

# Check for OpenMP availability
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    set(ETOPS_USE_OPENMP ON)
    set(EINSUM_IR_ENABLE_OPENMP ON)
else()
    set(ETOPS_USE_OPENMP OFF)
    set(EINSUM_IR_ENABLE_OPENMP OFF)
endif()

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------
# einsum_ir
# ------------------------------------------------------------
message(STATUS "Building einsum_ir from ../src/basic")
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../src/basic"
    einsum_ir_local
    EXCLUDE_FROM_ALL
)

# ------------------------------------------------------------
# Copy headers so that <einsum_ir/basic/...> resolves in-tree
# ------------------------------------------------------------
file(
    COPY
    "${CMAKE_CURRENT_LIST_DIR}/../src/basic"
    DESTINATION "${CMAKE_BINARY_DIR}/include/einsum_ir"
)

# ------------------------------------------------------------
# tpp_mlir and LLVM/MLIR setup
# ------------------------------------------------------------
message(STATUS "Including tpp_mlir from ${TPP_MLIR_DIR}")
message(STATUS "LLVM_DIR is set to ${LLVM_DIR}")

set(LLVM_INCLUDE_DIR ${LLVM_DIR}/../llvm/include)
set(LLVM_GENERATED_INCLUDE_DIR ${LLVM_DIR}/include)
set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR};${LLVM_GENERATED_INCLUDE_DIR}")
set(MLIR_INCLUDE_DIR ${LLVM_DIR}/../mlir/include)
set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_DIR}/tools/mlir/include)
set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
set(LLVM_BUILD_LIBRARY_DIR ${LLVM_DIR}/lib)
set(TPP_MLIR_API_DIR ${TPP_MLIR_DIR}/api)
set(TPP_MLIR_LIB_DIR ${TPP_MLIR_DIR}/build/lib)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_library(Einsum-API STATIC IMPORTED)
set_target_properties(Einsum-API PROPERTIES IMPORTED_LOCATION ${TPP_MLIR_LIB_DIR}/libEinsum-API.so)
set_target_properties(Einsum-API PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${TPP_MLIR_API_DIR})


# ------------------------------------------------------------
# Sources & target
# ------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_library(TensorOperation 
    STATIC 
    src/TensorOperation.cpp
    src/BackendFactory.cpp
    src/TppBackend.cpp
    src/TppMlirBackend.cpp)

target_link_libraries(TensorOperation PUBLIC einsum_ir Einsum-API)
target_include_directories(
    TensorOperation
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
)

if(ETOPS_USE_OPENMP)
    target_link_libraries(TensorOperation PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_etops_core MODULE src/bindings.cpp)
target_link_libraries(_etops_core PRIVATE TensorOperation)

if(APPLE)
    set_target_properties(
        _etops_core PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_etops_core"
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(
        _etops_core PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_etops_core"
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        LINK_FLAGS "-Wl,--enable-new-dtags"
    )
endif()

target_compile_features(_etops_core PUBLIC)

# ------------------------------------------------------------
# Install
# ------------------------------------------------------------
install(
    TARGETS einsum_ir
    LIBRARY DESTINATION etops
    ARCHIVE DESTINATION etops
    RUNTIME DESTINATION etops
)

install(
    TARGETS _etops_core
    LIBRARY DESTINATION etops
    RUNTIME DESTINATION etops
)