// Python binding (module `vla_cpp`) so the closed-loop benchmark can run the C++ policy.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "vla/policy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vla_cpp, m) {
  m.doc() = "C++ inference runtime for the VLA pick-and-place policy";
  py::class_<vla::Policy>(m, "Policy")
      .def(py::init<const std::string&>(), py::arg("weights_path"))
      .def_property_readonly("output_size", [](const vla::Policy& p) { return p.config().output_size(); })
      .def(
          "predict",
          [](vla::Policy& p,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> tokens,
             py::array_t<float, py::array::c_style | py::array::forcecast> proprio) {
            const auto& c = p.config();
            if (image.size() != static_cast<py::ssize_t>(c.img_h) * c.img_w * 3 ||
                tokens.size() != c.max_tokens || proprio.size() != c.proprio_dim)
              throw std::invalid_argument("input shapes do not match the model configuration");
            std::vector<float> out;
            {
              py::gil_scoped_release release;
              out = p.predict(image.data(), tokens.data(), proprio.data());
            }
            return py::array_t<float>(out.size(), out.data());
          },
          py::arg("image"), py::arg("tokens"), py::arg("proprio"),
          "Returns chunk*action_dim normalised actions (first action = first action_dim values).");
}
