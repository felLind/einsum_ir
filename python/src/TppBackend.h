#ifndef TPP_BACKEND_H
#define TPP_BACKEND_H

#include "BackendInterface.h"
#include <memory>
#include <einsum_ir/basic/unary/UnaryBackendTpp.h>
#include <einsum_ir/basic/unary/UnaryOptimizer.h>
#include <einsum_ir/basic/binary/ContractionBackendTpp.h>
#include <einsum_ir/basic/binary/ContractionOptimizer.h>

namespace einsum_ir {
namespace py {

/**
 * Backend-specific optimization parameters for TPP backend.
 */
struct TppOptimizationConfig {
    int64_t target_m            = 16;
    int64_t target_n            = 12;
    int64_t target_k            = 64;
    int64_t num_threads         = 0;        // Auto-detect
    bool    br_gemm_support     = true;
    bool    packed_gemm_support = true;
    bool    packing_support     = true;
    bool    sfc_support         = true;
    int64_t l2_cache_size       = 1048576;  // 1 MiB
};

/**
 * TPP Backend - wraps the original einsum_ir::basic backends directly
 */
class TppBackend : public BackendInterface {
private:
    op_type_t m_op_type = op_type_t::undefined;
    einsum_ir::basic::UnaryBackendTpp m_backend_unary;
    einsum_ir::basic::ContractionBackendTpp m_backend_binary;

public:
    TppBackend();
    virtual ~TppBackend() = default;

    virtual error_t setup(
        dtype_t dtype,
        prim_t prim_first,
        prim_t prim_main,
        prim_t prim_last,
        std::vector<dim_t> const & dim_types,
        std::vector<exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides
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

private:
    /**
     * Setup for unary operations.
     */
    error_t setup_unary(
        dtype_t dtype,
        prim_t prim_main,
        std::vector<exec_t> const & exec_types,
        std::vector<int64_t> const & dim_sizes,
        std::vector<int64_t> const & strides_in0,
        std::vector<int64_t> const & strides_out
    );

    /**
     * Setup for binary operations.
     */
   error_t setup_binary(
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
     * Implementation of optimize functionality.
     */
    std::tuple<
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
        TppOptimizationConfig const & optimization_config
    );

    /**
     * Optimize unary operations.
     */
    error_t optimize_unary(
        dtype_t dtype,
        prim_t & prim_main,
        std::vector<dim_t> & dim_types,
        std::vector<exec_t> & exec_types,
        std::vector<int64_t> & dim_sizes,
        std::vector<int64_t> & strides_in0,
        std::vector<int64_t> & strides_out,
        int64_t num_threads
    );

    /**
     * Optimize binary operations.
     */
    error_t optimize_binary(
        dtype_t dtype,
        prim_t & prim_main,
        std::vector<dim_t> & dim_types,
        std::vector<exec_t> & exec_types,
        std::vector<int64_t> & dim_sizes,
        std::vector<int64_t> & strides_in0,
        std::vector<int64_t> & strides_in1,
        std::vector<int64_t> & strides_out,
        std::vector<int64_t> & packing_in0,
        std::vector<int64_t> & packing_in1,
        int64_t target_m,
        int64_t target_n,
        int64_t target_k,
        int64_t num_threads[3],
        bool packed_gemm_support,
        bool br_gemm_support,
        bool packing_support,
        bool sfc_support,
        int64_t l2_cache_size
    );
};

} // namespace py
} // namespace einsum_ir

#endif // TPP_BACKEND_H