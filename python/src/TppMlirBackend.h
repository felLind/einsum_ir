#ifndef TPP_MLIR_BACKEND_H
#define TPP_MLIR_BACKEND_H

#include "BackendInterface.h"
#include "binary.h"
#include <memory>

namespace einsum_ir {
namespace py {

/**
 * TPP-MLIR Backend - wraps the BinaryContraction implementation
 */
class TppMlirBackend : public BackendInterface {
private:
    std::unique_ptr<mlir::einsum::BinaryContraction> m_binary_contraction;

    // Type conversion methods from TensorOperation types to binary types
    static mlir::einsum::data_t convert_dtype(dtype_t dtype);
    static mlir::einsum::kernel_t convert_prim(prim_t prim);
    static mlir::einsum::exec_t convert_exec(exec_t exec);
    static mlir::einsum::dim_t convert_dim(dim_t dim);

public:
    TppMlirBackend() = default;
    virtual ~TppMlirBackend() = default;

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
    ) override;

    virtual void execute(
        void const * tensor_in0,
        void const * tensor_in1,
        void * tensor_out
    ) override;

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
    ) override;

    virtual OptimizationConfig get_default_optimization_config() override;
};

} // namespace py
} // namespace einsum_ir

#endif // TPP_MLIR_BACKEND_H