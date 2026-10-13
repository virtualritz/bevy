use bevy_asset::{AssetId, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{render_resource::Buffer, sync_world::SyncToRenderWorld};
use bevy_transform::components::Transform;
use core::fmt;
use derive_more::derive::From;

/// Identifies the geometry a raytraced instance draws.
///
/// Most instances name a bevy [`Mesh`] asset. A host that builds vertices on the GPU names its
/// own geometry by a stable `u64` instead, so several placements of one geometry share a single
/// BLAS the way several placements of one asset do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RaytracingGeometryId {
    /// Geometry from a bevy [`Mesh`] asset.
    Mesh(AssetId<Mesh>),
    /// Geometry built on the GPU, named by its host.
    External(u64),
}

impl fmt::Display for RaytracingGeometryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mesh(id) => write!(formatter, "mesh {id:?}"),
            Self::External(id) => write!(formatter, "external {id}"),
        }
    }
}

/// A mesh component used for raytracing.
///
/// The mesh used in this component must have [`Mesh::enable_raytracing`] set to true,
/// use the following set of vertex attributes: `{POSITION, NORMAL, UV_0, TANGENT}`, use [`bevy_mesh::PrimitiveTopology::TriangleList`],
/// and use [`bevy_mesh::Indices::U32`].
///
/// The material used for this entity must be [`MeshMaterial3d<StandardMaterial>`].
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(MeshMaterial3d<StandardMaterial>, Transform, SyncToRenderWorld)]
pub struct RaytracingMesh3d(pub Handle<Mesh>);

/// Where one vertex attribute lives inside an [`ExternalMeshGeometry`]'s vertex buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalVertexAttribute {
    /// Byte offset of the attribute within the vertex buffer.
    pub byte_offset: u32,
    /// Byte step from one vertex to the next.
    pub byte_stride: u32,
}

impl ExternalVertexAttribute {
    /// A tight `f32x3` attribute starting at `byte_offset`.
    pub const fn packed_vec3(byte_offset: u32) -> Self {
        Self {
            byte_offset,
            byte_stride: 12,
        }
    }
}

/// Vertices and indices a host built on the GPU, for a raytraced instance.
///
/// The buffers are the host's own, on the same device the renderer runs on. Nothing is uploaded or
/// copied: [`super::RaytracingExternalMesh`] names them and `bevy_solari` builds the BLAS and
/// reads them for shading.
///
/// Attribute offsets and strides are in bytes, because the host decides its own vertex layout. The
/// renderer reads `f32` components, so every offset and stride must be a multiple of `4`.
#[derive(Clone, Debug)]
pub struct ExternalMeshGeometry {
    /// Stable identity for this geometry. Two instances naming the same id share one BLAS.
    pub id: u64,
    /// Vertices, in the layout the attributes below describe.
    pub vertex_buffer: Buffer,
    /// `u32` triangle-list indices.
    pub index_buffer: Buffer,
    /// Vertices in `vertex_buffer`.
    pub vertex_count: u32,
    /// Indices in `index_buffer`.
    pub index_count: u32,
    /// Positions, required. Offset `0` with the whole buffer's stride, as the BLAS reads them.
    pub position: ExternalVertexAttribute,
    /// Normals, required.
    pub normal: ExternalVertexAttribute,
    /// UVs, read as `(0, 0)` when absent.
    pub uv: Option<ExternalVertexAttribute>,
    /// Tangents, read as `(1, 0, 0, 1)` when absent.
    pub tangent: Option<ExternalVertexAttribute>,
}

/// A mesh component for raytracing whose geometry a host built on the GPU.
///
/// This is the sibling of [`RaytracingMesh3d`], in the same way [`bevy_mesh::Mesh`] assets sit
/// beside `MeshletMesh` assets. Exactly one of the two describes an entity.
///
/// The material used for this entity must be [`MeshMaterial3d<StandardMaterial>`].
///
/// The buffers must be created with [`bevy_render::render_resource::BufferUsages::BLAS_INPUT`], and
/// must stay alive for as long as an instance can reach them. `bevy_solari` keeps its own BLAS
/// alive past the frame that retired it, the same rule mesh slabs follow.
#[derive(Component, Clone, Debug)]
#[require(MeshMaterial3d<StandardMaterial>, Transform, SyncToRenderWorld)]
pub struct RaytracingExternalMesh(pub ExternalMeshGeometry);
