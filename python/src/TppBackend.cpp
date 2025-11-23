#include "TppBackend.h"
#include <set>
#include <pybind11/stl.h>

namespace py = pybind11;

#ifdef _OPENMP
#include <omp.h>
#endif

namespace einsum_ir {
namespace py {

// Helper functions moved from TensorOperation
TensorOperation::op_type_t determine_op_type(TensorOperation::prim_t prim_main) {
  switch (prim_main) {
    case TensorOperation::prim_t::zero:
    case TensorOperation::prim_t::copy:
      return TensorOperation::op_type_t::unary;
    case TensorOperation::prim_t::gemm:
    case TensorOperation::prim_t::brgemm:
      return TensorOperation::op_type_t::binary;
    default:
      return TensorOperation::op_type_t::undefined;
  }
}

int64_t get_num_threads(int64_t num_threads) {
  int64_t l_num_threads = num_threads;
#if defined(_OPENMP)
  if (l_num_threads <= 0) {
    l_num_threads = omp_get_max_threads();
  }
#else
  if (l_num_threads <= 0) {
    l_num_threads = 1;
  }
#endif
  return l_num_threads;
}

einsum_ir::basic::exec_t convert_exec_type(TensorOperation::exec_t exec_type) {
  switch (exec_type) {
    case TensorOperation::exec_t::seq:      return einsum_ir::basic::exec_t::SEQ;
    case TensorOperation::exec_t::prim:     return einsum_ir::basic::exec_t::PRIM;
    case TensorOperation::exec_t::shared:   return einsum_ir::basic::exec_t::OMP;
    case TensorOperation::exec_t::sfc:      return einsum_ir::basic::exec_t::SFC;
    default:                                return einsum_ir::basic::exec_t::UNDEFINED_EXECTYPE;
  }
}

einsum_ir::basic::dim_t convert_dim_type(TensorOperation::dim_t dim_type) {
  switch (dim_type) {
    case TensorOperation::dim_t::c:         return einsum_ir::basic::dim_t::C;
    case TensorOperation::dim_t::m:         return einsum_ir::basic::dim_t::M;
    case TensorOperation::dim_t::n:         return einsum_ir::basic::dim_t::N;    
    case TensorOperation::dim_t::k:         return einsum_ir::basic::dim_t::K;
    default:                                return einsum_ir::basic::dim_t::UNDEFINED_DIM;
  }
}

einsum_ir::basic::kernel_t convert_prim_to_kernel(TensorOperation::prim_t prim_type) {
  switch (prim_type) {
    case TensorOperation::prim_t::zero:     return einsum_ir::basic::kernel_t::ZERO;
    case TensorOperation::prim_t::copy:     return einsum_ir::basic::kernel_t::COPY;
    case TensorOperation::prim_t::relu:     return einsum_ir::basic::kernel_t::RELU;
    case TensorOperation::prim_t::gemm:     return einsum_ir::basic::kernel_t::MADD;
    case TensorOperation::prim_t::brgemm:   return einsum_ir::basic::kernel_t::BR_MADD;
    default:                                return einsum_ir::basic::kernel_t::UNDEFINED_KTYPE;
  }
}

// Reverse conversion functions
TensorOperation::exec_t convert_exec_type_back(einsum_ir::basic::exec_t exec_type) {
  switch (exec_type) {
    case einsum_ir::basic::exec_t::SEQ:  return TensorOperation::exec_t::seq;
    case einsum_ir::basic::exec_t::PRIM: return TensorOperation::exec_t::prim;
    case einsum_ir::basic::exec_t::OMP:  return TensorOperation::exec_t::shared;
    case einsum_ir::basic::exec_t::SFC:  return TensorOperation::exec_t::sfc;
    default:                             return TensorOperation::exec_t::undefined;
  }
}

TensorOperation::dim_t convert_dim_type_back(einsum_ir::basic::dim_t dim_type) {
  switch (dim_type) {
    case einsum_ir::basic::dim_t::C: return TensorOperation::dim_t::c;
    case einsum_ir::basic::dim_t::M: return TensorOperation::dim_t::m;
    case einsum_ir::basic::dim_t::N: return TensorOperation::dim_t::n;
    case einsum_ir::basic::dim_t::K: return TensorOperation::dim_t::k;
    default:                         return TensorOperation::dim_t::undefined;
  }
}

// Helper to create iter_properties from input parameters
std::vector<einsum_ir::basic::iter_property> create_iter_properties(
    std::vector<TensorOperation::dim_t> const & dim_types,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<int64_t> const & strides_in0,
    std::vector<int64_t> const & strides_in1,
    std::vector<int64_t> const & strides_out
) {
  std::vector<einsum_ir::basic::iter_property> iters;
  std::size_t l_num_iters = dim_types.size();
  
  for(std::size_t i = 0; i < l_num_iters; ++i) {
    einsum_ir::basic::iter_property l_iter;
    l_iter.dim_type       = convert_dim_type(dim_types[i]);
    l_iter.exec_type      = convert_exec_type(exec_types[i]);
    l_iter.size           = dim_sizes[i];
    l_iter.stride_left    = strides_in0[i];
    l_iter.stride_right   = strides_in1[i];
    l_iter.stride_out_aux = 0;
    l_iter.stride_out     = strides_out[i];
    l_iter.packing_stride_left  = 0;  // Will be set by optimizer
    l_iter.packing_stride_right = 0;  // Will be set by optimizer
    iters.push_back(l_iter);
  }
  
  return iters;
}

// Helper to update parameters from optimized iters
void update_parameters_from_iters(
  std::vector<einsum_ir::basic::iter_property> const & iters,
  std::vector<TensorOperation::dim_t> & dim_types,
  std::vector<TensorOperation::exec_t> & exec_types,
  std::vector<int64_t> & dim_sizes,
  std::vector<int64_t> & strides_in0,
  std::vector<int64_t> & strides_in1,
  std::vector<int64_t> & strides_out,
  std::vector<int64_t> & packing_in0,
  std::vector<int64_t> & packing_in1
) {
  size_t num_iters = iters.size();
  dim_types.clear();
  exec_types.clear();
  dim_sizes.clear();
  strides_in0.clear();
  strides_in1.clear();
  strides_out.clear();
  packing_in0.clear();
  packing_in1.clear();
  
  dim_types.reserve(num_iters);
  exec_types.reserve(num_iters);
  dim_sizes.reserve(num_iters);
  strides_in0.reserve(num_iters);
  strides_in1.reserve(num_iters);
  strides_out.reserve(num_iters);
  packing_in0.reserve(num_iters);
  packing_in1.reserve(num_iters);
  
  for (const auto & l_iter : iters) {
    dim_types.push_back(convert_dim_type_back(l_iter.dim_type));
    exec_types.push_back(convert_exec_type_back(l_iter.exec_type));
    dim_sizes.push_back(l_iter.size);
    strides_in0.push_back(l_iter.stride_left);
    strides_in1.push_back(l_iter.stride_right);
    strides_out.push_back(l_iter.stride_out);
    packing_in0.push_back(l_iter.packing_stride_left);
    packing_in1.push_back(l_iter.packing_stride_right);
  }
}

// Helper to get dtype size in bytes
int64_t dtype_to_num_bytes(TensorOperation::dtype_t dtype) {
  switch (dtype) {
    case TensorOperation::dtype_t::fp32: return sizeof(float);
    case TensorOperation::dtype_t::fp64: return sizeof(double);
    default:                             return 0; // Undefined or unsupported type
  }
}

void calculate_sfc_sizes(
  std::vector<TensorOperation::dim_t> const & dim_types,
  std::vector<TensorOperation::exec_t> const & exec_types,
  std::vector<int64_t> const & dim_sizes,
  int64_t & o_size_sfc_m,
  int64_t & o_size_sfc_n
) {
  o_size_sfc_m = 1;
  o_size_sfc_n = 1;

  for(std::size_t i = 0; i < dim_types.size(); i++) {
    if(exec_types[i] == TensorOperation::exec_t::sfc) {
      if(dim_types[i] == TensorOperation::dim_t::m) {
        o_size_sfc_m *= dim_sizes[i];
      } else if(dim_types[i] == TensorOperation::dim_t::n) {
        o_size_sfc_n *= dim_sizes[i];
      }
    }
  }
}

TppBackend::TppBackend() {
    // Default constructor - backends will be initialized in setup()
}

TensorOperation::error_t TppBackend::setup(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t prim_first,
    TensorOperation::prim_t prim_main,
    TensorOperation::prim_t prim_last,
    std::vector<TensorOperation::dim_t> const & dim_types,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides
) {
    m_op_type = determine_op_type(prim_main);

    if (m_op_type == TensorOperation::op_type_t::undefined) {
        return TensorOperation::error_t::compilation_failed;
    }

    // Validate stride dimensions: must have at least level 0
    if (strides.size() == 0) {
        return TensorOperation::error_t::invalid_stride_shape;
    }

    // Validate level 0 has correct number of tensors
    size_t expected_tensors = (m_op_type == TensorOperation::op_type_t::binary) ? 3 : 2;
    if (strides[0].size() != expected_tensors) {
        return TensorOperation::error_t::invalid_stride_shape;
    }

    // Extract level 0 strides from each tensor
    std::vector<int64_t> strides_in0, strides_in1, strides_out;

    // Validate and extract in0 strides
    if (strides[0][0].size() != dim_sizes.size()) {
        return TensorOperation::error_t::invalid_stride_shape;
    }
    strides_in0 = strides[0][0];

    if (m_op_type == TensorOperation::op_type_t::binary) {
        // Binary operation: validate and extract in1 and out strides
        if (strides[0][1].size() != dim_sizes.size()) {
            return TensorOperation::error_t::invalid_stride_shape;
        }
        strides_in1 = strides[0][1];

        if (strides[0][2].size() != dim_sizes.size()) {
            return TensorOperation::error_t::invalid_stride_shape;
        }
        strides_out = strides[0][2];

        return setup_binary(dtype, prim_first, prim_main, prim_last,
                           dim_types, exec_types, dim_sizes, strides);
    }
    else {
        // Unary operation: validate and extract out strides
        if (strides[0][1].size() != dim_sizes.size()) {
            return TensorOperation::error_t::invalid_stride_shape;
        }
        strides_out = strides[0][1];

        // Validate: prim_first and prim_last must be 'none' for unary
        if (prim_first != TensorOperation::prim_t::none || prim_last != TensorOperation::prim_t::none) {
            return TensorOperation::error_t::compilation_failed;
        }

        return setup_unary(dtype, prim_main, exec_types, dim_sizes, strides_in0, strides_out);
    }
}

void TppBackend::execute(
    void const * tensor_in0,
    void const * tensor_in1,
    void * tensor_out
) {
    if (m_op_type == TensorOperation::op_type_t::unary) {
        m_backend_unary.eval(tensor_in0, tensor_out);
    }
    else if (m_op_type == TensorOperation::op_type_t::binary) {
        m_backend_binary.contract(tensor_in0, tensor_in1, nullptr, tensor_out);
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
> TppBackend::optimize(
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
    // Parse the Python dict and create TppOptimizationConfig struct
    TppOptimizationConfig l_optimization_config;

    // Defaults are set in struct declaration

    // Valid keys for TPP backend
    std::set<std::string> valid_keys = {
        "target_m", "target_n", "target_k", "num_threads",
        "br_gemm_support", "packed_gemm_support", "packing_support",
        "sfc_support", "l2_cache_size"
    };

    try {
        // Check for unknown keys
        for (auto item : optimization_config) {
            std::string key = item.first.cast<std::string>();
            if (valid_keys.find(key) == valid_keys.end()) {
                std::vector<std::vector<std::vector<int64_t>>> empty_strides;
                return std::make_tuple(
                    TensorOperation::error_t::invalid_optimization_config,
                    dtype, prim_first, prim_main, prim_last,
                    dim_types, exec_types, dim_sizes, empty_strides
                );
            }
        }

        // Override defaults with provided values
        if (optimization_config.contains("target_m")) {
            l_optimization_config.target_m = optimization_config["target_m"].cast<int64_t>();
        }
        if (optimization_config.contains("target_n")) {
            l_optimization_config.target_n = optimization_config["target_n"].cast<int64_t>();
        }
        if (optimization_config.contains("target_k")) {
            l_optimization_config.target_k = optimization_config["target_k"].cast<int64_t>();
        }
        if (optimization_config.contains("num_threads")) {
            l_optimization_config.num_threads = optimization_config["num_threads"].cast<int64_t>();
        }
        if (optimization_config.contains("packed_gemm_support")) {
            l_optimization_config.packed_gemm_support = optimization_config["packed_gemm_support"].cast<bool>();
        }
        if (optimization_config.contains("br_gemm_support")) {
            l_optimization_config.br_gemm_support = optimization_config["br_gemm_support"].cast<bool>();
        }
        if (optimization_config.contains("packing_support")) {
            l_optimization_config.packing_support = optimization_config["packing_support"].cast<bool>();
        }
        if (optimization_config.contains("sfc_support")) {
            l_optimization_config.sfc_support = optimization_config["sfc_support"].cast<bool>();
        }
        if (optimization_config.contains("l2_cache_size")) {
            l_optimization_config.l2_cache_size = optimization_config["l2_cache_size"].cast<int64_t>();
        }
    } catch (...) {
        // Type casting failed
        std::vector<std::vector<std::vector<int64_t>>> empty_strides;
        return std::make_tuple(
            TensorOperation::error_t::invalid_optimization_config,
            dtype, prim_first, prim_main, prim_last,
            dim_types, exec_types, dim_sizes, empty_strides
        );
    }

    // Call our own optimize implementation
    return optimize_impl(
        dtype, prim_first, prim_main, prim_last,
        dim_types, exec_types, dim_sizes, strides,
        l_optimization_config
    );
}

py::dict TppBackend::get_default_optimization_config() {
    // Get the struct with defaults
    TppOptimizationConfig l_config;

    // Convert to Python dict
    py::dict result;
    result["target_m"]            = l_config.target_m;
    result["target_n"]            = l_config.target_n;
    result["target_k"]            = l_config.target_k;
    result["num_threads"]         = l_config.num_threads;
    result["packed_gemm_support"] = l_config.packed_gemm_support;
    result["br_gemm_support"]     = l_config.br_gemm_support;
    result["packing_support"]     = l_config.packing_support;
    result["sfc_support"]         = l_config.sfc_support;
    result["l2_cache_size"]       = l_config.l2_cache_size;

    return result;
}

TensorOperation::error_t TppBackend::setup_unary(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t prim_main,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<int64_t> const & strides_in0,
    std::vector<int64_t> const & strides_out
) {
    // Convert data types
    einsum_ir::basic::data_t l_dtype_in = einsum_ir::basic::data_t::UNDEFINED_DTYPE;
    switch (dtype) {
        case TensorOperation::dtype_t::fp32: l_dtype_in = einsum_ir::basic::data_t::FP32; break;
        case TensorOperation::dtype_t::fp64: l_dtype_in = einsum_ir::basic::data_t::FP64; break;
        default: l_dtype_in = einsum_ir::basic::data_t::UNDEFINED_DTYPE; break;
    }
    einsum_ir::basic::data_t l_dtype_comp = l_dtype_in;
    einsum_ir::basic::data_t l_dtype_out = l_dtype_in;

    // Convert kernel type
    einsum_ir::basic::kernel_t l_ktype_main = convert_prim_to_kernel(prim_main);

    // Convert exec types
    std::vector<einsum_ir::basic::exec_t> l_exec_types;
    l_exec_types.reserve(exec_types.size());
    for (const auto & l_exec_type : exec_types) {
        l_exec_types.push_back(convert_exec_type(l_exec_type));
    }

    // Auto-detect number of threads
    int64_t l_num_threads = get_num_threads(0);

    // Initialize unary backend
    m_backend_unary.init(l_exec_types, dim_sizes, strides_in0, strides_out,
                        l_dtype_in, l_dtype_comp, l_dtype_out,
                        l_ktype_main, l_num_threads);

    // Compile backend
    einsum_ir::basic::err_t l_err = m_backend_unary.compile();
    if (l_err != einsum_ir::basic::err_t::SUCCESS) {
        return TensorOperation::error_t::compilation_failed;
    }

    return TensorOperation::error_t::success;
}

TensorOperation::error_t TppBackend::setup_binary(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t prim_first,
    TensorOperation::prim_t prim_main,
    TensorOperation::prim_t prim_last,
    std::vector<TensorOperation::dim_t> const & dim_types,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides
) {
    // Auto-detect number of threads
    int64_t l_num_threads[3] = {1, 1, 1};
    l_num_threads[0] = get_num_threads(0);

    // Calculate SFC dimension sizes from configuration
    int64_t l_size_sfc_m = 1;
    int64_t l_size_sfc_n = 1;
    calculate_sfc_sizes(dim_types, exec_types, dim_sizes, l_size_sfc_m, l_size_sfc_n);

    // Distribute threads for SFC dimensions if present
    if(l_size_sfc_m > 1 || l_size_sfc_n > 1) {
        einsum_ir::basic::ContractionOptimizer::set_num_threads_sfc(
            l_size_sfc_m,
            l_size_sfc_n,
            &l_num_threads[0],
            &l_num_threads[1],
            &l_num_threads[2]
        );
    }

    // Extract level 0 strides (already validated in setup())
    std::vector<int64_t> l_strides_in0 = strides[0][0];
    std::vector<int64_t> l_strides_in1 = strides[0][1];
    std::vector<int64_t> l_strides_out = strides[0][2];

    // Extract level 1 (packing) strides if available
    std::vector<int64_t> l_packing_strides_left;
    std::vector<int64_t> l_packing_strides_right;

    if (strides.size() > 1 && strides[1].size() == 3) {
        if (strides[1][0].size() == dim_sizes.size()) {
            l_packing_strides_left = strides[1][0];
        } else {
            l_packing_strides_left = std::vector<int64_t>(dim_sizes.size(), 0);
        }

        if (strides[1][1].size() == dim_sizes.size()) {
            l_packing_strides_right = strides[1][1];
        } else {
            l_packing_strides_right = std::vector<int64_t>(dim_sizes.size(), 0);
        }
    } else {
        l_packing_strides_left = std::vector<int64_t>(dim_sizes.size(), 0);
        l_packing_strides_right = std::vector<int64_t>(dim_sizes.size(), 0);
    }

    // Convert to backend enums
    std::vector<einsum_ir::basic::dim_t> l_dim_types;
    std::vector<einsum_ir::basic::exec_t> l_exec_types;

    l_dim_types.reserve(dim_types.size());
    for (const auto & l_dim_type : dim_types) {
        l_dim_types.push_back(convert_dim_type(l_dim_type));
    }

    l_exec_types.reserve(exec_types.size());
    for (const auto & l_exec_type : exec_types) {
        l_exec_types.push_back(convert_exec_type(l_exec_type));
    }

    // Convert data types
    einsum_ir::basic::data_t l_dtype_left = einsum_ir::basic::data_t::UNDEFINED_DTYPE;
    switch (dtype) {
        case TensorOperation::dtype_t::fp32: l_dtype_left = einsum_ir::basic::data_t::FP32; break;
        case TensorOperation::dtype_t::fp64: l_dtype_left = einsum_ir::basic::data_t::FP64; break;
        default: l_dtype_left = einsum_ir::basic::data_t::UNDEFINED_DTYPE; break;
    }
    einsum_ir::basic::data_t l_dtype_right = l_dtype_left;
    einsum_ir::basic::data_t l_dtype_comp = l_dtype_left;
    einsum_ir::basic::data_t l_dtype_out = l_dtype_left;

    // Convert kernel types
    einsum_ir::basic::kernel_t l_ktype_first = convert_prim_to_kernel(prim_first);
    einsum_ir::basic::kernel_t l_ktype_main = convert_prim_to_kernel(prim_main);
    einsum_ir::basic::kernel_t l_ktype_last = convert_prim_to_kernel(prim_last);

    // dummy strides for auxiliary output tensor
    std::vector<int64_t> l_strides_out_aux(dim_sizes.size(), 0);

    // Initialize backend
    m_backend_binary.init(l_dim_types,
                         l_exec_types,
                         dim_sizes,
                         l_strides_in0,
                         l_strides_in1,
                         l_strides_out_aux,
                         l_strides_out,
                         l_packing_strides_left,
                         l_packing_strides_right,
                         l_dtype_left,
                         l_dtype_right,
                         l_dtype_comp,
                         l_dtype_out,
                         l_ktype_first,
                         l_ktype_main,
                         l_ktype_last,
                         l_num_threads[0],
                         l_num_threads[1],
                         l_num_threads[2],
                         nullptr);

    // Compile backend
    einsum_ir::basic::err_t l_err = m_backend_binary.compile();
    if (l_err != einsum_ir::basic::err_t::SUCCESS) {
        return TensorOperation::error_t::compilation_failed;
    }

    return TensorOperation::error_t::success;
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
> TppBackend::optimize_impl(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t prim_first,
    TensorOperation::prim_t prim_main,
    TensorOperation::prim_t prim_last,
    std::vector<TensorOperation::dim_t> const & dim_types,
    std::vector<TensorOperation::exec_t> const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<std::vector<std::vector<int64_t>>> const & strides,
    TppOptimizationConfig const & optimization_config
) {
    std::vector<std::vector<std::vector<int64_t>>> l_empty_strides;

    // Extract optimization parameters from config
    int64_t l_target_m            = optimization_config.target_m;
    int64_t l_target_n            = optimization_config.target_n;
    int64_t l_target_k            = optimization_config.target_k;
    bool    l_packed_gemm_support = optimization_config.packed_gemm_support;
    bool    l_br_gemm_support     = optimization_config.br_gemm_support;
    bool    l_packing_support     = optimization_config.packing_support;
    bool    l_sfc_support         = optimization_config.sfc_support;
    int64_t l_l2_cache_size       = optimization_config.l2_cache_size;

    int64_t l_num_threads[3] = {1, 1, 1};
    l_num_threads[0] = get_num_threads(optimization_config.num_threads);

    TensorOperation::op_type_t l_op_type = determine_op_type(prim_main);

    // Validate stride dimensions based on operation type
    size_t l_expected_tensors = (l_op_type == TensorOperation::op_type_t::binary) ? 3 : 2;

    if (l_op_type == TensorOperation::op_type_t::undefined) {
        return std::make_tuple(TensorOperation::error_t::compilation_failed, dtype, prim_first,
                               prim_main, prim_last, dim_types, exec_types,
                               dim_sizes, l_empty_strides);
    }

    // Validate stride dimensions: must have at least level 0
    if (strides.size() == 0) {
        return std::make_tuple(TensorOperation::error_t::invalid_stride_shape, dtype, prim_first,
                               prim_main, prim_last, dim_types, exec_types,
                               dim_sizes, l_empty_strides);
    }

    // Validate level 0 has correct number of tensors
    if (strides[0].size() != l_expected_tensors) {
        return std::make_tuple(TensorOperation::error_t::invalid_stride_shape, dtype, prim_first,
                               prim_main, prim_last, dim_types, exec_types,
                               dim_sizes, l_empty_strides);
    }

    // Validate that level 0 has proper dimensions for each tensor
    for (size_t t = 0; t < l_expected_tensors; ++t) {
        if (strides[0][t].size() != dim_sizes.size()) {
            return std::make_tuple(TensorOperation::error_t::invalid_stride_shape, dtype, prim_first,
                                   prim_main, prim_last, dim_types, exec_types,
                                   dim_sizes, l_empty_strides);
        }
    }

    // Extract level 0 strides
    std::vector<int64_t> l_strides_in0 = strides[0][0];
    std::vector<int64_t> l_strides_in1 = (l_op_type == TensorOperation::op_type_t::binary) ? strides[0][1]
                                                                                        : std::vector<int64_t>(dim_sizes.size(), 0);
    std::vector<int64_t> l_strides_out = (l_op_type == TensorOperation::op_type_t::binary) ? strides[0][2] : strides[0][1];

    // Make copies of input parameters for optimization
    std::vector<TensorOperation::dim_t> l_opt_dim_types = dim_types;
    std::vector<TensorOperation::exec_t> l_opt_exec_types = exec_types;
    std::vector<int64_t> l_opt_dim_sizes = dim_sizes;
    std::vector<int64_t> l_opt_strides_in0 = l_strides_in0;
    std::vector<int64_t> l_opt_strides_in1 = l_strides_in1;
    std::vector<int64_t> l_opt_strides_out = l_strides_out;
    std::vector<int64_t> l_opt_packing_in0;
    std::vector<int64_t> l_opt_packing_in1;
    TensorOperation::prim_t l_opt_prim_main = prim_main;

    TensorOperation::error_t l_err = TensorOperation::error_t::success;

    if (l_op_type == TensorOperation::op_type_t::unary) {
        // Unary optimization
        l_err = optimize_unary(dtype, l_opt_prim_main, l_opt_dim_types, l_opt_exec_types,
                               l_opt_dim_sizes, l_opt_strides_in0, l_opt_strides_out,
                               l_num_threads[0]);
    } else {
        // Binary optimization
        l_err = optimize_binary(dtype, l_opt_prim_main, l_opt_dim_types, l_opt_exec_types,
                                l_opt_dim_sizes, l_opt_strides_in0, l_opt_strides_in1,
                                l_opt_strides_out, l_opt_packing_in0, l_opt_packing_in1,
                                l_target_m, l_target_n, l_target_k, l_num_threads,
                                l_packed_gemm_support, l_br_gemm_support, l_packing_support,
                                l_sfc_support, l_l2_cache_size);
    }

    if (l_err != TensorOperation::error_t::success) {
        return std::make_tuple(l_err, dtype, prim_first, l_opt_prim_main, prim_last,
                               l_opt_dim_types, l_opt_exec_types, l_opt_dim_sizes, l_empty_strides);
    }

    // Build output 3D strides with [LEVEL][TENSOR][DIMENSION] order
    std::vector<std::vector<std::vector<int64_t>>> opt_strides;

    if (l_op_type == TensorOperation::op_type_t::binary) {
        // Check if packing strides are all zero
        bool l_has_packing = false;
        for (size_t i = 0; i < l_opt_packing_in0.size() && !l_has_packing; ++i) {
            if (l_opt_packing_in0[i] != 0 || l_opt_packing_in1[i] != 0) {
                l_has_packing = true;
            }
        }

        if (l_has_packing) {
            // Return with 2 levels: level 0 (primary) and level 1 (packing)
            opt_strides = {
                {l_opt_strides_in0, l_opt_strides_in1, l_opt_strides_out},  // level 0
                {l_opt_packing_in0, l_opt_packing_in1, std::vector<int64_t>(l_opt_strides_out.size(), 0)}  // level 1 (out has no packing)
            };
        } else {
            // Return with single level only
            opt_strides = {
                {l_opt_strides_in0, l_opt_strides_in1, l_opt_strides_out}  // level 0
            };
        }
    } else {
        // Unary: single level only
        opt_strides = {
            {l_opt_strides_in0, l_opt_strides_out}  // level 0
        };
    }

    return std::make_tuple(l_err, dtype, prim_first, l_opt_prim_main, prim_last,
                           l_opt_dim_types, l_opt_exec_types, l_opt_dim_sizes, opt_strides);
}

TensorOperation::error_t TppBackend::optimize_unary(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t & prim_main,
    std::vector<TensorOperation::dim_t> & dim_types,
    std::vector<TensorOperation::exec_t> & exec_types,
    std::vector<int64_t> & dim_sizes,
    std::vector<int64_t> & strides_in0,
    std::vector<int64_t> & strides_out,
    int64_t num_threads
) {
    // Create iter_properties for unary operation
    std::vector<einsum_ir::basic::iter_property> l_iters;
    l_iters.reserve(dim_sizes.size());

    for (size_t i = 0; i < dim_sizes.size(); ++i) {
        einsum_ir::basic::iter_property l_iter;
        l_iter.dim_type       = convert_dim_type(dim_types[i]);
        l_iter.exec_type      = convert_exec_type(exec_types[i]);
        l_iter.size           = dim_sizes[i];
        l_iter.stride_left    = strides_in0[i];
        l_iter.stride_right   = 0;  // unused for unary
        l_iter.stride_out_aux = 0;  // unused for unary
        l_iter.stride_out     = strides_out[i];
        l_iter.packing_stride_left  = 0;  // unused for unary
        l_iter.packing_stride_right = 0;  // unused for unary
        l_iters.push_back(l_iter);
    }

    // Initialize optimizer (scalar_optim = false)
    einsum_ir::basic::UnaryOptimizer l_optimizer;
    l_optimizer.init(&l_iters, num_threads, false);

    // Run optimization
    einsum_ir::basic::err_t l_err = l_optimizer.optimize();
    if (l_err != einsum_ir::basic::err_t::SUCCESS) {
        return TensorOperation::error_t::compilation_failed;
    }

    // Update parameters from optimized iters
    dim_types.clear();
    exec_types.clear();
    dim_sizes.clear();
    strides_in0.clear();
    strides_out.clear();

    dim_types.reserve(l_iters.size());
    exec_types.reserve(l_iters.size());
    dim_sizes.reserve(l_iters.size());
    strides_in0.reserve(l_iters.size());
    strides_out.reserve(l_iters.size());

    for (const auto & l_iter : l_iters) {
        dim_types.push_back(convert_dim_type_back(l_iter.dim_type));
        exec_types.push_back(convert_exec_type_back(l_iter.exec_type));
        dim_sizes.push_back(l_iter.size);
        strides_in0.push_back(l_iter.stride_left);
        strides_out.push_back(l_iter.stride_out);
    }

    return TensorOperation::error_t::success;
}

TensorOperation::error_t TppBackend::optimize_binary(
    TensorOperation::dtype_t dtype,
    TensorOperation::prim_t & prim_main,
    std::vector<TensorOperation::dim_t> & dim_types,
    std::vector<TensorOperation::exec_t> & exec_types,
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
) {
    // Create iter_properties from input parameters
    std::vector<einsum_ir::basic::iter_property> l_iters = create_iter_properties(
        dim_types,
        exec_types,
        dim_sizes,
        strides_in0,
        strides_in1,
        strides_out
    );
    
    // Convert main primitive type to kernel type  
    einsum_ir::basic::kernel_t l_kernel_main = convert_prim_to_kernel(prim_main);
    int64_t l_num_bytes = dtype_to_num_bytes(dtype);
    einsum_ir::basic::packed_gemm_t l_packed_gemm_support =
        packed_gemm_support ? einsum_ir::basic::packed_gemm_t::ALL_STRIDE_ONE
                            : einsum_ir::basic::packed_gemm_t::NONE;

    // Initialize optimizer
    einsum_ir::basic::ContractionOptimizer l_optimizer;
    l_optimizer.init(&l_iters,
                     &l_kernel_main,
                     target_m,
                     target_n,
                     target_k,
                     sfc_support,
                     br_gemm_support,
                     packing_support,
                     l_packed_gemm_support,
                     l_num_bytes,
                     l2_cache_size,
                     &num_threads[0],
                     &num_threads[1],
                     &num_threads[2]);
    
    // Run optimization
    l_optimizer.optimize();
    
    // Extract optimized parameters including packing strides
    update_parameters_from_iters(
        l_iters,
        dim_types,
        exec_types,
        dim_sizes,
        strides_in0,
        strides_in1,
        strides_out,
        packing_in0,
        packing_in1
    );
    
    // Update primitive type if changed
    if (l_kernel_main == einsum_ir::basic::kernel_t::BR_MADD) {
        prim_main = TensorOperation::prim_t::brgemm;
    } else if (l_kernel_main == einsum_ir::basic::kernel_t::MADD) {
        prim_main = TensorOperation::prim_t::gemm;
    }
    
    return TensorOperation::error_t::success;
}

} // namespace py
} // namespace einsum_ir