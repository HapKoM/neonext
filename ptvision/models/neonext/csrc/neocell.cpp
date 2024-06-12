#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <stdexcept>

using namespace torch::indexing;

void assign_blocks_forward(
  torch::Tensor matr, torch::Tensor weight,
  int k_h, int k_w, int shift,
  int channel_start, int channel_end,
  int full_height, int full_width) {
  if (shift != 0) {
    // Fill upper-left square shift*shift with lower-right part of kernel
    matr.index_put_(
      {
        Slice(channel_start, channel_end, 1),
        Slice(0, shift, 1),
        Slice(0, shift, 1)
      },
      weight.index({
        Slice(channel_start, channel_end, 1),
        Slice(k_h - shift, k_h, 1),
        Slice(k_w - shift, k_w, 1)
      })
    );
  }

  int repeats = std::max({
    std::ceil((float) (full_height - shift) / k_h),
    std::ceil((float) (full_width  - shift) / k_w)
  });
  for (int i = 0; i < repeats; ++i) {
    int rem_h = k_h - std::max(0, (i + 1) * k_h - (full_height - shift));
    int rem_w = k_w - std::max(0, (i + 1) * k_w - (full_width  - shift));
    matr.index_put_(
      {
        Slice(channel_start, channel_end, 1),
        Slice(shift + i * k_h, shift + (i + 1) * k_h, 1),
        Slice(shift + i * k_w, shift + (i + 1) * k_w, 1)
      },
      weight.index({
        Slice(channel_start, channel_end, 1),
        Slice(0, rem_h, 1),
        Slice(0, rem_w, 1)
      })
    );
  }
}

void sum_blocks_backward(
  torch::Tensor grad_matr, torch::Tensor grad,
  int k_h, int k_w, int shift,
  int channel_start, int channel_end,
  int sub_channel_start, int sub_channel_end,
  int full_height, int full_width) {
  if (shift != 0) {
    // Fill lower-right part of the kernel gradients with the upper-left
    // square shift*shift of the full matrix
    grad.index_put_(
      {
        Slice(sub_channel_start, sub_channel_end, 1),
        Slice(k_h - shift, k_h, 1),
        Slice(k_w - shift, k_w, 1)
      },
      grad_matr.index({
        Slice(channel_start + sub_channel_start, channel_start + sub_channel_end, 1),
        Slice(0, shift, 1),
        Slice(0, shift, 1)
      })
    );
  }

  int repeats = std::max({
    std::ceil((float) (full_height - shift) / k_h),
    std::ceil((float) (full_width  - shift) / k_w)
  });
  for (int i = 0; i < repeats; ++i) {
    int rem_h = k_h - std::max(0, (i + 1) * k_h - (full_height - shift));
    int rem_w = k_w - std::max(0, (i + 1) * k_w - (full_width  - shift));
    auto grad_matr_part = grad_matr.index({
      Slice(channel_start + sub_channel_start, channel_start + sub_channel_end, 1),
      Slice(shift + i * k_h, shift + (i + 1) * k_h, 1),
      Slice(shift + i * k_w, shift + (i + 1) * k_w, 1)
    });
    grad.index({
      Slice(sub_channel_start, sub_channel_end, 1),
      Slice(0, rem_h, 1),
      Slice(0, rem_w, 1)
    }) += grad_matr_part;
  }
}

/*
 * get_matrices_forward - construction of the NeoCell block-diagonal matrices
 *   given the learnable weights weight_a and weight_b
 */
std::vector<at::Tensor> get_matrices_forward(
  std::vector<torch::Tensor> weight_a,
  std::vector<torch::Tensor> weight_b,
  std::vector< std::map<std::string, int> > channel_specs,
  int a_full_height, int a_full_width,
  int b_full_height, int b_full_width) {

  std::vector<torch::Tensor> matr_a_list;
  std::vector<torch::Tensor> matr_b_list;

  size_t j = 0;

  for (auto &spec: channel_specs) {
    int h_in, h_out, w_in, w_out, channels, shift_a, shift_b = -1;
    try {
      // kernel size
      if (spec.find("h_in") != spec.end()) {
        h_in  = spec["h_in"];
        h_out = spec["h_out"];
        w_in  = spec["w_in"];
        w_out = spec["w_out"];
      } else if (spec.find("kernel_a") != spec.end()) {
        h_in = h_out = spec["kernel_a"];
        w_in = w_out = spec["kernel_b"];
      } else {
        h_in = h_out = w_in = w_out = spec["kernel"];
      }
      // channels
      channels = spec["channels"];
      // shift
      if (spec.find("shift_a") != spec.end()) {
        shift_a = spec["shift_a"];
        shift_b = spec["shift_b"];
      } else {
        shift_a = shift_b = spec["shift"];
      }
    }
    catch (...) {
      throw std::invalid_argument("NeoCell.forward: Invalid channel specification");
    }

    auto dtype = weight_a[j].dtype();
    auto options = torch::TensorOptions()
      .dtype(dtype)
      .device(weight_a[j].device());

    torch::Tensor matr_a = torch::zeros({channels, a_full_height, a_full_width}, options);
    torch::Tensor matr_b = torch::zeros({channels, b_full_height, b_full_width}, options);

    if ((shift_a > 0 && (h_out != h_in)) || (shift_b > 0 && (w_out != w_in))) {
      std::cerr << shift_a << " " << shift_b << std::endl;
      std::cerr << h_in << " " << h_out << std::endl;
      std::cerr << w_in << " " << w_out << std::endl;
      throw std::invalid_argument(
        "NeoCell.forward: shift is supported only for squared kernels");
    }
    if ((h_out <= shift_a) || (w_out <= shift_b)) {
      throw std::invalid_argument(
        "NeoCell.forward: shift must be smaller than kernel size");
    }

    int num_a = (shift_a > 0) ? (h_out - 1) / shift_a + 1 : 1;
    int num_b = (shift_b > 0) ? (w_out - 1) / shift_b + 1 : 1;
    int shift_a_group_size = channels / num_a;
    int shift_b_group_size = channels / num_b;

    // matrix A
    for (int s_a = 0; s_a < num_a; ++s_a) {
      int cur_shift = s_a * shift_a;
      int channel_start = s_a * shift_a_group_size;
      int channel_end = (s_a + 1) * shift_a_group_size;
      if (s_a >= num_a - 1) {
        channel_end = channels;
      }
      assign_blocks_forward(
        matr_a, weight_a[j], h_out, h_in, cur_shift,
        channel_start, channel_end, a_full_height, a_full_width
      );
    }

    // matrix B
    for (int s_b = 0; s_b < num_b; ++s_b) {
      int cur_shift = s_b * shift_b;
      int channel_start = s_b * shift_b_group_size;
      int channel_end = (s_b + 1) * shift_b_group_size;
      if (s_b == num_b - 1) {
        channel_end = channels;
      }
      assign_blocks_forward(
        matr_b, weight_b[j], w_in, w_out, cur_shift,
        channel_start, channel_end, b_full_height, b_full_width
      );
    }

    matr_a_list.push_back(matr_a);
    matr_b_list.push_back(matr_b);
    j++;
  }

  at::Tensor matr_a_all = at::cat(matr_a_list, 0);
  at::Tensor matr_b_all = at::cat(matr_b_list, 0);
  return {matr_a_all, matr_b_all};
}

/*
 * get_matrices_backward - computing the gradients of the block-diagonalization
 *   operation get_matrices_forward
 */
std::vector<std::vector<at::Tensor>> get_matrices_backward(
  torch::Tensor grad_matr_a,
  torch::Tensor grad_matr_b,
  std::vector< std::map<std::string, int> > channel_specs,
  int a_full_height, int a_full_width,
  int b_full_height, int b_full_width) {

  std::vector<torch::Tensor> weight_a_grad;
  std::vector<torch::Tensor> weight_b_grad;

  size_t j = 0;
  size_t channel_start = 0; // Start channel of this kernel's group

  for (auto &spec: channel_specs) {
    int h_in, h_out, w_in, w_out, channels, shift_a, shift_b = -1;
    try {
      // kernel size
      if (spec.find("h_in") != spec.end()) {
        h_in  = spec["h_in"];
        h_out = spec["h_out"];
        w_in  = spec["w_in"];
        w_out = spec["w_out"];
      } else if (spec.find("kernel_a") != spec.end()) {
        h_in = h_out = spec["kernel_a"];
        w_in = w_out = spec["kernel_b"];
      } else {
        h_in = h_out = w_in = w_out = spec["kernel"];
      }
      // channels
      channels = spec["channels"];
      // shift
      if (spec.find("shift_a") != spec.end()) {
        shift_a = spec["shift_a"];
        shift_b = spec["shift_b"];
      } else {
        shift_a = shift_b = spec["shift"];
      }
    }
    catch (...) {
      throw std::invalid_argument("NeoCell.backward: Invalid channel specification");
    }

    auto channel_end = channel_start + channels; // End channel of this kernel's group
    auto options = torch::TensorOptions()
      .dtype(grad_matr_a.dtype()) // force torch::kFloat32 for reduction ???
      .device(grad_matr_a.device());

    // Initialize gradients with zeros
    auto a_grad = torch::zeros({channels, h_out, h_in}, options);
    auto b_grad = torch::zeros({channels, w_in, w_out}, options);

    if ((shift_a > 0 && h_out != h_in) || (shift_b > 0 && w_out != w_in)) {
      throw std::invalid_argument(
        "NeoCell.backward: shift is supported only for squared kernels");
    }

    if ((h_out <= shift_a) || (w_out <= shift_b)) {
      throw std::invalid_argument(
        "NeoCell.backward: shift must be smaller than kernel size");
    }

    int num_a = (shift_a > 0) ? (h_out - 1) / shift_a + 1 : 1;
    int num_b = (shift_b > 0) ? (w_out - 1) / shift_b + 1 : 1;
    int shift_a_group_size = channels / num_a;
    int shift_b_group_size = channels / num_b;

    // Matrix A
    for (int s_a = 0; s_a < num_a; ++s_a) {
      int cur_shift = s_a * shift_a;
      int sub_channel_start = s_a * shift_a_group_size; // Start sub channel of this shift split group
      int sub_channel_end = (s_a + 1) * shift_a_group_size; // End sub channel of this shift split group
      if (s_a == num_a - 1) {
        sub_channel_end = channels;
      }
      sum_blocks_backward(
        grad_matr_a, a_grad, h_out, h_in, cur_shift,
        channel_start, channel_end, sub_channel_start, sub_channel_end,
        a_full_height, a_full_width
      );
    }

    // Matrix B
    for (int s_b = 0; s_b < num_b; ++s_b) {
      int cur_shift = s_b * shift_b;
      int sub_channel_start = s_b * shift_b_group_size; // Start sub channel of this shift split group
      int sub_channel_end = (s_b + 1) * shift_b_group_size; // End sub channel of this shift split group
      if (s_b == num_b - 1) {
        sub_channel_end = channels;
      }
      sum_blocks_backward(
        grad_matr_b, b_grad, w_in, w_out, cur_shift,
        channel_start, channel_end, sub_channel_start, sub_channel_end,
        b_full_height, b_full_width
      );
    }

    weight_a_grad.push_back(a_grad);
    weight_b_grad.push_back(b_grad);

    j++;
    channel_start = channel_end;
  }

  return {weight_a_grad, weight_b_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_matrices_forward", &get_matrices_forward, "NeoCell get matrices forward");
  m.def("get_matrices_backward", &get_matrices_backward, "NeoCell get matrices backward");
}
