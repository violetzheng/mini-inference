#include "layers/linear_layer.h"

namespace mini_inference::layers
{

    mini_inference::tensor::Tensor forward(const LinearLayer &layer, const mini_inference::tensor::Tensor &input)
    {
        return std::visit([&input](const auto &proj)
                           { return proj.forward(input); },
                           layer);
    }

    std::size_t in_features(const LinearLayer &layer)
    {
        return std::visit([](const auto &proj)
                           { return proj.in_features(); },
                           layer);
    }

    std::size_t out_features(const LinearLayer &layer)
    {
        return std::visit([](const auto &proj)
                           { return proj.out_features(); },
                           layer);
    }

} // namespace mini_inference::layers
