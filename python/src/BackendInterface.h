#ifndef BACKEND_INTERFACE_H
#define BACKEND_INTERFACE_H

#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <memory>

namespace einsum_ir {
namespace py {

// Forward declarations
class TensorOperation;

/// operation type
enum class op_type_t : uint32_t {
    binary    = 0,
    unary     = 1,
    undefined = 99
};

/// execution type
enum class exec_t : uint32_t {
    seq       = 0, 
    prim      = 1,
    shared    = 2,
    sfc       = 3,
    undefined = 99
};

/// primitive type
enum class prim_t : uint32_t {
    none      =  0,
    zero      =  1,
    copy      =  2,
    relu      =  3,
    gemm      =  4,
    brgemm    =  5,
    undefined = 99
};

/// dimension type
enum class dim_t : uint32_t {
    c         = 0, 
    m         = 1, 
    n         = 2, 
    k         = 3, 
    undefined = 99
};

/// data type
enum class dtype_t : uint32_t {
    fp32 = 0,
    fp64 = 1
};

/// error codes
enum class error_t : int32_t {
    success                     = 0,
    compilation_failed          = 1,
    invalid_stride_shape        = 2,
    invalid_optimization_config = 3
};

/**
 * Backend-specific optimization parameters.
 */
struct OptimizationConfig {
    int64_t target_m            = 0;
    int64_t target_n            = 0;
    int64_t target_k            = 0;
    int64_t num_threads         = 0;
    bool    br_gemm_support     = false;
    bool    packed_gemm_support = false;
    bool    packing_support     = false;
    bool    sfc_support         = false;
    int64_t l2_cache_size       = 0;
};

struct CompilerConfig {
    std::string feature;
    unsigned optLevel;
    bool computeGrid;
    std::vector<unsigned> grid;
    bool debugLog = false;
}; 

/**
 * Abstract base class for all backends
 */
class BackendInterface {
public:
    virtual ~BackendInterface() = default;

    virtual error_t setup(
        dtype_t dtype,
        prim_t prim_first,
        prim_t prim_main,
        prim_t prim_last,
        std::vector<dim_t> const & dim_types,
        std::vector<exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides,
        CompilerConfig compilerConfig = CompilerConfig()
    ) = 0;

    virtual void execute(
        void const * tensor_in0,
        void const * tensor_in1,
        void * tensor_out
    ) = 0;

    virtual std::tuple<
        error_t,
        dtype_t,
        prim_t,
        prim_t,
        prim_t,
        std::vector<dim_t>,
        std::vector<exec_t>,
        std::vector<int64_t>,
        std::vector<std::vector<std::vector<int64_t>>>
    > optimize(
        dtype_t dtype,
        prim_t prim_first,
        prim_t prim_main,
        prim_t prim_last,
        std::vector<dim_t> const & dim_types,
        std::vector<exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides,
        OptimizationConfig const & optimization_config
    ) = 0;

    virtual OptimizationConfig get_default_optimization_config() = 0;
};

/**
 * Factory for creating backend instances
 */
class BackendFactory {
public:
    static std::unique_ptr<BackendInterface> create_backend(std::string const & backend_name);
    static bool is_backend_supported(std::string const & backend_name);
};

} // namespace py
} // namespace einsum_ir

#endif // BACKEND_INTERFACE_H