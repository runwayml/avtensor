/// PyTensor type that wraps a `tch::Tensor` for use in Python.
///
/// Adapted from https://github.com/LaurentMazare/tch-rs/blob/main/pyo3-tch/src/lib.rs
/// we're using an adapted version in order to work around a PyO3 compatibility issue that causes a linker
/// error when running tests. See [1].
///
/// [1]: https://pyo3.rs/main/faq#i-cant-run-cargo-test-or-i-cant-build-in-a-cargo-workspace-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror
use pyo3::prelude::*;

use pyo3::exceptions::{PyTypeError, PyValueError};
pub use tch;

pub struct PyTensor(pub tch::Tensor);

impl std::ops::Deref for PyTensor {
    type Target = tch::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn wrap_tch_err(err: tch::TchError) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) };
        tensor
            .map_err(wrap_tch_err)?
            .ok_or_else(|| {
                let type_ = ob.get_type();
                PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
            })
            .map(PyTensor)
    }
}

impl<'py> IntoPyObject<'py> for PyTensor {
    type Output = Bound<'py, Self::Target>;
    type Target = PyAny;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        let v = self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        );
        Ok(v.into_pyobject(py)?)
    }
}
