#include "TppMlirBackend.h"

namespace einsum_ir {
namespace py {

// Type conversion methods - these are the mapper methods in the backend implementation
mlir::einsum::data_t TppMlirBackend::convert_dtype(dtype_t dtype) {
    switch (dtype) {
        case dtype_t::fp32: return mlir::einsum::data_t::FP32;
        case dtype_t::fp64: return mlir::einsum::data_t::FP32; // Binary backend only supports FP32
        default: return mlir::einsum::data_t::FP32;
    }
}

mlir::einsum::kernel_t TppMlirBackend::convert_prim(prim_t prim) {
    switch (prim) {
        case prim_t::none: return mlir::einsum::kernel_t::NONE;
        case prim_t::zero: return mlir::einsum::kernel_t::ZERO;
        case prim_t::copy: return mlir::einsum::kernel_t::COPY;
        case prim_t::relu: return mlir::einsum::kernel_t::RELU;
        case prim_t::gemm: return mlir::einsum::kernel_t::GEMM;
        case prim_t::brgemm: return mlir::einsum::kernel_t::BRGEMM;
        default: return mlir::einsum::kernel_t::UNDEFINED_KERNEL;
    }
}

mlir::einsum::exec_t TppMlirBackend::convert_exec(exec_t exec) {
    switch (exec) {
        case exec_t::seq: return mlir::einsum::exec_t::SEQ;
        case exec_t::prim: return mlir::einsum::exec_t::PRIM;
        case exec_t::shared: return mlir::einsum::exec_t::SHARED;
        case exec_t::sfc: return mlir::einsum::exec_t::SFC;
        default: return mlir::einsum::exec_t::UNDEFINED_EXEC;
    }
}

mlir::einsum::dim_t TppMlirBackend::convert_dim(dim_t dim) {
    switch (dim) {
        case dim_t::c: return mlir::einsum::dim_t::C;
        case dim_t::m: return mlir::einsum::dim_t::M;
        case dim_t::n: return mlir::einsum::dim_t::N;
        case dim_t::k: return mlir::einsum::dim_t::K;
        default: return mlir::einsum::dim_t::UNDEFINED_DIM;
    }
}

error_t TppMlirBackend::setup(
    dtype_t dtype,
    prim_t prim_first,
    prim_t prim_main,
    prim_t prim_last,
    std::vector<dim_t> const & dim_types,
    std::vector<exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides,
    CompilerConfig compilerConfig
) {
    // Create binary configuration using mapper methods
    mlir::einsum::iter_config config;
    
    config.d_type = convert_dtype(dtype);
    config.prim_first_touch = convert_prim(prim_first);
    config.prim_main = convert_prim(prim_main);
    config.prim_last_touch = convert_prim(prim_last);
    
    // Convert dimension types
    for (const auto& dt : dim_types) {
        config.dim_types.push_back(convert_dim(dt));
    }
    
    // Convert execution types
    for (const auto& et : exec_types) {
        config.exec_types.push_back(convert_exec(et));
    }
    
    config.sizes = dim_sizes;
    config.strides = strides;

    mlir::einsum::CompilerOptions options;
    options.feature = compilerConfig.feature;
    options.optLevel = compilerConfig.optLevel;
    options.parallelTaskGrid = std::vector<unsigned>(
        compilerConfig.grid.begin(),
        compilerConfig.grid.end()
    );
    options.debugLog = compilerConfig.debugLog;

    try {
        m_binary_contraction = std::make_unique<mlir::einsum::BinaryContraction>(config, options);
        return error_t::success;
    } catch (...) {
        return error_t::compilation_failed;
    }
}

void TppMlirBackend::execute(
    void const * tensor_in0,
    void const * tensor_in1,
    void * tensor_out
) {
    if (m_binary_contraction) {
        // Cast to float for binary backend
        const float* in0 = static_cast<const float*>(tensor_in0);
        const float* in1 = static_cast<const float*>(tensor_in1);
        float* out = static_cast<float*>(tensor_out);
        
        m_binary_contraction->execute(in0, in1, out);
    }
}

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
> TppMlirBackend::optimize(
    dtype_t dtype,
    prim_t prim_first,
    prim_t prim_main,
    prim_t prim_last,
    std::vector<dim_t> const & dim_types,
    std::vector<exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides,
    OptimizationConfig const & optimization_config
) {
    // TPP-MLIR backend doesn't support optimization, return original configuration
    return std::make_tuple(
        error_t::success,
        dtype, prim_first, prim_main, prim_last,
        dim_types, exec_types, dim_sizes, strides
    );
}

OptimizationConfig TppMlirBackend::get_default_optimization_config() {
    // TPP-MLIR backend doesn't have optimization parameters
    return OptimizationConfig();
}

} // namespace py
} // namespace einsum_ir