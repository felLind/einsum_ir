#include "BackendInterface.h"
#include "TppBackend.h"
#include "TppMlirBackend.h"

namespace einsum_ir {
namespace py {

std::unique_ptr<BackendInterface> BackendFactory::create_backend(std::string const & backend_name) {
    if (backend_name == "tpp") {
        return std::make_unique<TppBackend>();
    }
    else if (backend_name == "tpp-mlir") {
        return std::make_unique<TppMlirBackend>();
    }
    else {
        return nullptr;
    }
}

bool BackendFactory::is_backend_supported(std::string const & backend_name) {
    return (backend_name == "tpp" || backend_name == "tpp-mlir");
}

} // namespace py
} // namespace einsum_ir