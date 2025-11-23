#include "TppMlirBackend.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace einsum_ir {
namespace py {

// Type conversion methods - these are the mapper methods in the backend implementation
mlir::einsum::data_t TppMlirBackend::convert_dtype(TensorOperation::dtype_t dtype) {
    switch (dtype) {
        case TensorOperation::dtype_t::fp32: return mlir::einsum::data_t::FP32;
        case TensorOperation::dtype_t::fp64: return mlir::einsum::data_t::FP32; // Binary backend only supports FP32
        default: return mlir::einsum::data_t::FP32;
    }
}

mlir::einsum::kernel_t TppMlirBackend::convert_prim(TensorOperation::prim_t prim) {
    switch (prim) {
        case TensorOperation::prim_t::none: return mlir::einsum::kernel_t::NONE;
        case TensorOperation::prim_t::zero: return mlir::einsum::kernel_t::ZERO;
        case TensorOperation::prim_t::copy: return mlir::einsum::kernel_t::COPY;
        case TensorOperation::prim_t::relu: return mlir::einsum::kernel_t::RELU;
        case TensorOperation::prim_t::gemm: return mlir::einsum::kernel_t::GEMM;
        case TensorOperation::prim_t::brgemm: return mlir::einsum::kernel_t::BRGEMM;
        default: return mlir::einsum::kernel_t::UNDEFINED_KERNEL;
    }
}

mlir::einsum::exec_t TppMlirBackend::convert_exec(TensorOperation::exec_t exec) {
    switch (exec) {
        case TensorOperation::exec_t::seq: return mlir::einsum::exec_t::SEQ;
        case TensorOperation::exec_t::prim: return mlir::einsum::exec_t::PRIM;
        case TensorOperation::exec_t::shared: return mlir::einsum::exec_t::SHARED;
        case TensorOperation::exec_t::sfc: return mlir::einsum::exec_t::SFC;
        default: return mlir::einsum::exec_t::UNDEFINED_EXEC;
    }
}

mlir::einsum::dim_t TppMlirBackend::convert_dim(TensorOperation::dim_t dim) {
    switch (dim) {
        case TensorOperation::dim_t::c: return mlir::einsum::dim_t::C;
        case TensorOperation::dim_t::m: return mlir::einsum::dim_t::M;
        case TensorOperation::dim_t::n: return mlir::einsum::dim_t::N;
        case TensorOperation::dim_t::k: return mlir::einsum::dim_t::K;
        default: return mlir::einsum::dim_t::UNDEFINED_DIM;
    }
}

TensorOperation::error_t TppMlirBackend::setup(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t prim_first,
    TensorOperation::prim_t prim_main,
    TensorOperation::prim_t prim_last,
    std::vector<TensorOperation::dim_t> const & dim_types,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides
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

    try {
        m_binary_contraction = std::make_unique<mlir::einsum::BinaryContraction>(config);
        return TensorOperation::error_t::success;
    } catch (...) {
        return TensorOperation::error_t::compilation_failed;
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
    TensorOperation::error_t,
    TensorOperation::dtype_t,
    TensorOperation::prim_t,
    TensorOperation::prim_t,
    TensorOperation::prim_t,
    std::vector<TensorOperation::dim_t>,
    std::vector<TensorOperation::exec_t>,
    std::vector<int64_t>,
    std::vector<std::vector<std::vector<int64_t>>>
> TppMlirBackend::optimize(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t prim_first,
    TensorOperation::prim_t prim_main,
    TensorOperation::prim_t prim_last,
    std::vector<TensorOperation::dim_t> const & dim_types,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides,
    py::dict const & optimization_config
) {
    // TPP-MLIR backend doesn't support optimization, return original configuration
    return std::make_tuple(
        TensorOperation::error_t::success,
        dtype, prim_first, prim_main, prim_last,
        dim_types, exec_types, dim_sizes, strides
    );
}

py::dict TppMlirBackend::get_default_optimization_config() {
    // TPP-MLIR backend doesn't have optimization parameters
    return py::dict();
}

} // namespace py
} // namespace einsum_ir