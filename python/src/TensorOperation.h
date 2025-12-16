#ifndef EINSUM_IR_PY_TENSOR_OPERATION_H
#define EINSUM_IR_PY_TENSOR_OPERATION_H

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#include "BackendInterface.h"

// Forward declarations for backend system
namespace einsum_ir {
  namespace py {
    class BackendInterface;
  }
}

namespace einsum_ir {
  namespace py {
    class TensorOperation;
  }
}

class einsum_ir::py::TensorOperation {
  public:
    // Backend system
    std::unique_ptr<BackendInterface> m_backend_interface;
    std::string m_backend_name;

    /**
     * Setup for a binary tensor contraction or a unary tensor operation with backend selection.
     *
     * @param backend    Backend identifier ("tpp" or "tpp-mlir"). Defaults to "tpp" for backwards compatibility.
     * @param dtype      Datatype of all tensor elements.
     * @param prim_first Type of the first touch primitive.
     * @param prim_main  Type of the main primitive (determines operation type).
     * @param prim_last  Type of the last touch primitive.
     * @param dim_types  Dimension types.
     * @param exec_types Execution type of the dimensions (prim, seq, shared, or sfc).
     * @param dim_sizes  Sizes of the dimensions.
     * @param strides    3D stride tensor: [LEVEL][TENSOR][DIMENSION]
     *                   - LEVEL: 0=primary layout, 1=packing, 2+=reserved
     *                   - TENSOR: 0=in0, 1=in1, 2=out (binary) or 0=in, 1=out (unary)
     *                   - DIMENSION: dimension index
     * @return           Appropriate error code.
     **/
    error_t setup(
      std::string const & backend,
      dtype_t dtype,
      prim_t prim_first,
      prim_t prim_main,
      prim_t prim_last,
      std::vector<dim_t> const & dim_types,
      std::vector<exec_t> const & exec_types,
      std::vector<int64_t> const & dim_sizes,
      std::vector<std::vector<std::vector<int64_t>>> const & strides,
      CompilerConfig compilerConfig = CompilerConfig()
    );

    /**
     * Setup for a binary tensor contraction or a unary tensor operation (legacy method for backwards compatibility).
     * Uses "tpp" backend by default.
     *
     * @param dtype      Datatype of all tensor elements.
     * @param prim_first Type of the first touch primitive.
     * @param prim_main  Type of the main primitive (determines operation type).
     * @param prim_last  Type of the last touch primitive.
     * @param dim_types  Dimension types.
     * @param exec_types Execution type of the dimensions (prim, seq, shared, or sfc).
     * @param dim_sizes  Sizes of the dimensions.
     * @param strides    3D stride tensor: [LEVEL][TENSOR][DIMENSION]
     * @return           Appropriate error code.
     **/
    error_t setup(
      dtype_t dtype,
      prim_t prim_first,
      prim_t prim_main,
      prim_t prim_last,
      std::vector<dim_t> const & dim_types,
      std::vector<exec_t> const & exec_types,
      std::vector<int64_t> const & dim_sizes,
      std::vector<std::vector<std::vector<int64_t>>> const & strides
    );

    /**
     * Execute the tensor operation.
     *
     * @param tensor_in0 First input tensor.
     * @param tensor_in1 Second input tensor (use nullptr if unary).
     * @param tensor_out Output tensor.
     **/
    void execute( void const * tensor_in0,
                  void const * tensor_in1,
                  void       * tensor_out );

    /**
     * Optimizes a tensor operation configuration with backend selection.
     *
     * The operation type is automatically determined from prim_main.
     * For binary operations, may add packing levels if optimization determines it beneficial.
     * For unary operations, returns single level.
     *
     * @param backend             Backend identifier ("tpp" or "tpp-mlir").
     * @param dtype               Datatype of all tensor elements.
     * @param prim_first          Type of the first touch primitive.
     * @param prim_main           Type of the main primitive.
     * @param prim_last           Type of the last touch primitive.
     * @param dim_types           Dimension types.
     * @param exec_types          Execution type of the dimensions.
     * @param dim_sizes           Sizes of the dimensions.
     * @param strides             3D stride tensor: [LEVEL][TENSOR][DIMENSION]
     * @param optimization_config Backend-specific optimization parameters.
     * @return                    Tuple of (error, dtype, prim_first, prim_main, prim_last,
     *                            dim_types, exec_types, dim_sizes, optimized_strides).
     **/
    static std::tuple<
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
      std::string const & backend,
      dtype_t dtype,
      prim_t prim_first,
      prim_t prim_main,
      prim_t prim_last,
      std::vector<dim_t> const & dim_types,
      std::vector<exec_t> const & exec_types,
      std::vector<int64_t> const & dim_sizes,
      std::vector<std::vector<std::vector<int64_t>>> const & strides,
      einsum_ir::py::OptimizationConfig const & optimization_config
    );



    /**
     * Get default optimization configuration for a backend.
     *
     * @param backend Backend identifier ("tpp" or "tpp-mlir").
     * @return        Default optimization parameters for the backend.
     **/
    static einsum_ir::py::OptimizationConfig get_default_optimization_config(std::string const & backend);
};

#endif
