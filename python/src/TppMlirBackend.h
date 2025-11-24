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
    static mlir::einsum::data_t convert_dtype(TensorOperation::dtype_t dtype);
    static mlir::einsum::kernel_t convert_prim(TensorOperation::prim_t prim);
    static mlir::einsum::exec_t convert_exec(TensorOperation::exec_t exec);
    static mlir::einsum::dim_t convert_dim(TensorOperation::dim_t dim);

public:
    TppMlirBackend() = default;
    virtual ~TppMlirBackend() = default;

    virtual TensorOperation::error_t setup(
        TensorOperation::dtype_t dtype,
        TensorOperation::prim_t prim_first,
        TensorOperation::prim_t prim_main,
        TensorOperation::prim_t prim_last,
        std::vector<TensorOperation::dim_t> const & dim_types,
        std::vector<TensorOperation::exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides
    ) override;

    virtual void execute(
        void const * tensor_in0,
        void const * tensor_in1,
        void * tensor_out
    ) override;

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
        OptimizationConfig const & optimization_config
    ) override;

    virtual OptimizationConfig get_default_optimization_config() override;
};

} // namespace py
} // namespace einsum_ir

#endif // TPP_MLIR_BACKEND_H