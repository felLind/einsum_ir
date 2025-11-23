#ifndef BACKEND_INTERFACE_H
#define BACKEND_INTERFACE_H

#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsum_ir {
namespace py {

// Forward declarations
class TensorOperation;

/**
 * Abstract base class for all backends
 */
class BackendInterface {
public:
    virtual ~BackendInterface() = default;

    virtual TensorOperation::error_t setup(
        TensorOperation::dtype_t dtype,
        TensorOperation::prim_t prim_first,
        TensorOperation::prim_t prim_main,
        TensorOperation::prim_t prim_last,
        std::vector<TensorOperation::dim_t> const & dim_types,
        std::vector<TensorOperation::exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides
    ) = 0;

    virtual void execute(
        void const * tensor_in0,
        void const * tensor_in1,
        void * tensor_out
    ) = 0;

    virtual std::tuple<
        TensorOperation::error_t,
        TensorOperation::dtype_t,
        TensorOperation::prim_t,
        TensorOperation::prim_t,
        TensorOperation::prim_t,
        std::vector<TensorOperation::dim_t>,
        std::vector<TensorOperation::exec_t>,
        std::vector<int64_t>,
        std::vector<std::vector<std::vector<int64_t>>>
    > optimize(
        TensorOperation::dtype_t dtype,
        TensorOperation::prim_t prim_first,
        TensorOperation::prim_t prim_main,
        TensorOperation::prim_t prim_last,
        std::vector<TensorOperation::dim_t> const & dim_types,
        std::vector<TensorOperation::exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides,
        py::dict const & optimization_config
    ) = 0;

    virtual py::dict get_default_optimization_config() = 0;
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