#[macro_use]
mod options;

#[macro_use]
#[cfg(feature = "vision")]
mod detection_common_impl;

pub(crate) use options::*;

/// Load `buf` into a wasi-nn graph for the backend reported by `model_resource`.
pub(crate) fn build_graph(
    model_resource: &dyn crate::model::ModelResourceTrait,
    device: crate::Device,
    buf: &[u8],
) -> Result<crate::Graph, crate::Error> {
    Ok(crate::GraphBuilder::new(model_resource.model_backend(), device).build_from_bytes([buf])?)
}
