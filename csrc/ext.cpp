/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "./pulsar/global.h" // Include before <torch/extension.h>.
#include <torch/extension.h>
// clang-format on
#include "./pulsar/pytorch/renderer.h"
#include "./pulsar/pytorch/tensor_util.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Pulsar.
  // Pulsar not enabled on AMD.
#ifdef PULSAR_LOGGING_ENABLED
  c10::ShowLogInfoToStderr();
#endif
  py::class_<
      pulsar::pytorch::Renderer,
      std::shared_ptr<pulsar::pytorch::Renderer>>(m, "PulsarRenderer")
      .def(py::init<
           const uint&,
           const uint&,
           const uint&,
           const bool&,
           const bool&,
           const float&,
           const uint&,
           const uint&>())
      .def(
          "__eq__",
          [](const pulsar::pytorch::Renderer& a,
             const pulsar::pytorch::Renderer& b) { return a == b; },
          py::is_operator())
      .def(
          "__ne__",
          [](const pulsar::pytorch::Renderer& a,
             const pulsar::pytorch::Renderer& b) { return !(a == b); },
          py::is_operator())
      .def(
          "__repr__",
          [](const pulsar::pytorch::Renderer& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def(
          "forward",
          &pulsar::pytorch::Renderer::forward,
          py::arg("vert_pos"),
          py::arg("vert_col"),
          py::arg("vert_radii"),

          py::arg("cam_pos"),
          py::arg("pixel_0_0_center"),
          py::arg("pixel_vec_x"),
          py::arg("pixel_vec_y"),
          py::arg("focal_length"),
          py::arg("principal_point_offsets"),

          py::arg("gamma"),
          py::arg("max_depth"),
          py::arg("min_depth") /* = 0.f*/,
          py::arg("bg_col") /* = std::nullopt not exposed properly in
                               pytorch 1.1. */
          ,
          py::arg("opacity") /* = std::nullopt ... */,
          py::arg("percent_allowed_difference") = 0.01f,
          py::arg("max_n_hits") = MAX_UINT,
          py::arg("mode") = 0)
      .def("backward", &pulsar::pytorch::Renderer::backward)
      .def_property(
          "device_tracker",
          [](const pulsar::pytorch::Renderer& self) {
            return self.device_tracker;
          },
          [](pulsar::pytorch::Renderer& self, const torch::Tensor& val) {
            self.device_tracker = val;
          })
      .def_property_readonly("width", &pulsar::pytorch::Renderer::width)
      .def_property_readonly("height", &pulsar::pytorch::Renderer::height)
      .def_property_readonly(
          "max_num_balls", &pulsar::pytorch::Renderer::max_num_balls)
      .def_property_readonly(
          "orthogonal", &pulsar::pytorch::Renderer::orthogonal)
      .def_property_readonly(
          "right_handed", &pulsar::pytorch::Renderer::right_handed)
      .def_property_readonly("n_track", &pulsar::pytorch::Renderer::n_track);
  m.def(
      "pulsar_sphere_ids_from_result_info_nograd",
      &pulsar::pytorch::sphere_ids_from_result_info_nograd);
  // Constants.
  m.attr("EPS") = py::float_(EPS);
  m.attr("MAX_FLOAT") = py::float_(MAX_FLOAT);
  m.attr("MAX_INT") = py::int_(MAX_INT);
  m.attr("MAX_UINT") = py::int_(MAX_UINT);
  m.attr("MAX_USHORT") = py::int_(MAX_USHORT);
  m.attr("PULSAR_MAX_GRAD_SPHERES") = py::int_(MAX_GRAD_SPHERES);
}
