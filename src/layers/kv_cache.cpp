#include "layers/kv_cache.h"

#include <stdexcept>
#include <string>

namespace mini_inference::layers
{

    namespace
    {

        void validate_rows(const mini_inference::tensor::Tensor &t, std::size_t hidden_dim, const char *name)
        {
            if (t.rank() != 2)
            {
                throw std::invalid_argument(std::string(name) + " must be a 2D tensor");
            }
            if (t.shape()[1] != hidden_dim)
            {
                throw std::invalid_argument(std::string(name) + " feature count does not match cache hidden_dim");
            }
        }

    } // namespace

    KvCache::KvCache(std::size_t max_seq_len, std::size_t hidden_dim)
        : max_seq_len_(max_seq_len),
          hidden_dim_(hidden_dim)
    {
        if (max_seq_len_ == 0)
        {
            throw std::invalid_argument("max_seq_len must be greater than zero");
        }
        if (hidden_dim_ == 0)
        {
            throw std::invalid_argument("hidden_dim must be greater than zero");
        }
        k_storage_.assign(max_seq_len_ * hidden_dim_, 0.0f);
        v_storage_.assign(max_seq_len_ * hidden_dim_, 0.0f);
    }

    std::size_t KvCache::length() const
    {
        return length_;
    }

    std::size_t KvCache::max_seq_len() const
    {
        return max_seq_len_;
    }

    std::size_t KvCache::hidden_dim() const
    {
        return hidden_dim_;
    }

    void KvCache::append(const mini_inference::tensor::Tensor &k, const mini_inference::tensor::Tensor &v)
    {
        validate_rows(k, hidden_dim_, "kv cache key rows");
        validate_rows(v, hidden_dim_, "kv cache value rows");
        if (k.shape()[0] != v.shape()[0])
        {
            throw std::invalid_argument("kv cache key and value row counts must match");
        }

        const std::size_t num_new = k.shape()[0];
        if (length_ + num_new > max_seq_len_)
        {
            throw std::out_of_range("kv cache append exceeds max_seq_len");
        }

        const std::size_t offset = length_ * hidden_dim_;
        const std::vector<float> &k_values = k.values();
        const std::vector<float> &v_values = v.values();
        for (std::size_t i = 0; i < num_new * hidden_dim_; ++i)
        {
            k_storage_[offset + i] = k_values[i];
            v_storage_[offset + i] = v_values[i];
        }

        length_ += num_new;
    }

    mini_inference::tensor::Tensor KvCache::keys() const
    {
        std::vector<float> values(k_storage_.begin(), k_storage_.begin() + length_ * hidden_dim_);
        return mini_inference::tensor::Tensor({length_, hidden_dim_}, std::move(values));
    }

    mini_inference::tensor::Tensor KvCache::values() const
    {
        std::vector<float> values(v_storage_.begin(), v_storage_.begin() + length_ * hidden_dim_);
        return mini_inference::tensor::Tensor({length_, hidden_dim_}, std::move(values));
    }

    void KvCache::reset()
    {
        length_ = 0;
    }

} // namespace mini_inference::layers
