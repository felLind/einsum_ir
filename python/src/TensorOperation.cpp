#include "TensorOperation.h"
#include "BackendInterface.h"
#include <cstdint>
#include <tuple>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declaration for BackendFactory
namespace einsum_ir {
namespace py {
class BackendFactory {
public:
    static std::unique_ptr<BackendInterface> create_backend(std::string const & backend_name);
    static bool is_backend_supported(std::string const & backend_name);
};
}
}







// New setup method with backend support
einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::setup(
  std::string const & backend,
  dtype_t dtype,
  prim_t prim_first,
  prim_t prim_main,
  prim_t prim_last,
  std::vector<dim_t> const & dim_types,
  std::vector<exec_t> const & exec_types,
  std::vector<int64_t> const & dim_sizes,
  std::vector<std::vector<std::vector<int64_t>>> const & strides
) {
  // Create the appropriate backend
  m_backend_interface = BackendFactory::create_backend(backend);
  if (!m_backend_interface) {
    return error_t::compilation_failed; // Unsupported backend
  }
  
  m_backend_name = backend;
  
  // Forward to backend
  return m_backend_interface->setup(dtype, prim_first, prim_main, prim_last,
                                   dim_types, exec_types, dim_sizes, strides);
}

// Legacy setup method for backwards compatibility (uses "tpp" backend)
einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::setup(
  dtype_t dtype,
  prim_t prim_first,
  prim_t prim_main,
  prim_t prim_last,
  std::vector<dim_t> const & dim_types,
  std::vector<exec_t> const & exec_types,
  std::vector<int64_t> const & dim_sizes,
  std::vector<std::vector<std::vector<int64_t>>> const & strides
) {
  // Use "tpp" backend by default for backwards compatibility
  return setup("tpp", dtype, prim_first, prim_main, prim_last,
               dim_types, exec_types, dim_sizes, strides);
}





void einsum_ir::py::TensorOperation::execute( void const * tensor_in0,
                                              void const * tensor_in1,
                                              void       * tensor_out) {
  // Use backend interface
  if (m_backend_interface) {
    m_backend_interface->execute(tensor_in0, tensor_in1, tensor_out);
  }
}



// New optimize method with backend support
std::tuple<
  einsum_ir::py::TensorOperation::error_t,
  einsum_ir::py::TensorOperation::dtype_t,
  einsum_ir::py::TensorOperation::prim_t,
  einsum_ir::py::TensorOperation::prim_t,
  einsum_ir::py::TensorOperation::prim_t,
  std::vector<einsum_ir::py::TensorOperation::dim_t>,
  std::vector<einsum_ir::py::TensorOperation::exec_t>,
  std::vector<int64_t>,
  std::vector<std::vector<std::vector<int64_t>>>
> einsum_ir::py::TensorOperation::optimize(
  std::string const & backend,
  dtype_t dtype,
  prim_t prim_first,
  prim_t prim_main,
  prim_t prim_last,
  std::vector<dim_t> const & dim_types,
  std::vector<exec_t> const & exec_types,
  std::vector<int64_t> const & dim_sizes,
  std::vector<std::vector<std::vector<int64_t>>> const & strides,
  py::dict const & optimization_config
) {
  // Create temporary backend for optimization
  auto temp_backend = BackendFactory::create_backend(backend);
  if (!temp_backend) {
    // Unsupported backend
    std::vector<std::vector<std::vector<int64_t>>> empty_strides;
    return std::make_tuple(
      error_t::compilation_failed,
      dtype, prim_first, prim_main, prim_last,
      dim_types, exec_types, dim_sizes, empty_strides
    );
  }
  
  // Forward to backend
  return temp_backend->optimize(dtype, prim_first, prim_main, prim_last,
                               dim_types, exec_types, dim_sizes, strides,
                               optimization_config);
}

py::dict einsum_ir::py::TensorOperation::get_default_optimization_config(std::string const & backend) {
  // Create temporary backend for getting default config
  auto temp_backend = BackendFactory::create_backend(backend);
  if (!temp_backend) {
    return py::dict(); // Empty dict for unsupported backend
  }
  
  return temp_backend->get_default_optimization_config();
}
