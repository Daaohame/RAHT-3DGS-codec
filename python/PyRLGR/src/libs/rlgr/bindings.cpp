#include <stdio.h>
#include <stdint.h>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <vector>
// #include "file.h"
#include "membuf.h"

namespace py = pybind11;

// PYBIND11_MODULE(rlgr, m) {
//   m.doc() = "pybind11 rlgr";

//     py::class_<file>(m, "file")
//       .def(py::init<char*, uint_least8_t>())
//       .def("rlgrRead", [](file &f, size_t N, uint_least8_t flagSigned) {
//           std::vector<int64_t> seq(N, 0);
//           auto runtime = f.rlgrRead(seq.data(), N, flagSigned);
//           return std::make_pair(runtime.count(), seq);
//       })
//       .def("rlgrWrite", [](file &f, std::vector<int64_t> seq, uint_least8_t flagSigned) {
//           auto runtime = f.rlgrWrite(seq.data(), seq.size(), flagSigned);
//           return runtime.count();
//       })
//       .def("grRead", &file::grRead)
//       .def("grWrite", &file::grWrite)
//       .def("openError", &file::openError)
//       .def("close", &file::close);
// }


PYBIND11_MODULE(rlgr, m) {
  m.doc() = "pybind11 rlgr (bitstream buffer version)";

    // Bind std::vector<uint8_t> for automatic conversion to Python list
    py::bind_vector<std::vector<uint8_t>>(m, "ByteVector");

    py::class_<membuf>(m, "membuf")
      .def(py::init<>()) // Bind the write constructor
      .def(py::init<const std::vector<uint8_t>&>()) // Bind the read constructor
      .def("get_buffer", &membuf::get_buffer)
      .def("rlgrRead", [](membuf &m, size_t N, uint_least8_t flagSigned) {
          std::vector<int64_t> seq(N, 0);
          auto runtime = m.rlgrRead(seq.data(), N, flagSigned);
          return std::make_pair(runtime.count(), seq);
      })
      .def("rlgrWrite", [](membuf &m, std::vector<int64_t> seq, uint_least8_t flagSigned) {
          auto runtime = m.rlgrWrite(seq.data(), seq.size(), flagSigned);
          return runtime.count();
      })
      .def("grRead", &membuf::grRead)
      .def("grWrite", &membuf::grWrite)
      .def("close", &membuf::close)
      .def("buffer_size", &membuf::buffer_size);
}