use super::{
    extract::ExtractedExternalMeshes, types::ExternalVertexAttribute, ExternalMeshGeometry,
    RaytracingGeometryId,
};
use alloc::collections::VecDeque;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_mesh::{Indices, Mesh};
use bevy_platform::collections::HashMap;
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    mesh::{allocator::MeshAllocator, RenderMesh},
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};
use tracing::warn;

/// After compacting this many vertices worth of meshes per frame, no further BLAS will be compacted.
/// Lower this number to distribute the work across more frames.
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 400_000;

/// Under the `wgpu_hal` build path, we need to manage BLAS lifetimes ourselves.
/// Since solari keeps both current and previous frame TLAS's around, only after
/// two TLAS builds since we marked it for deletion is it safe to delete a BLAS.
const TLAS_BUILDS_BEFORE_DELETION_ALLOWED: usize = 2;

/// Byte step between the vertices [`MeshAllocator`] stores for a mesh with the attribute set
/// `is_mesh_raytracing_compatible` requires: `POSITION`, `NORMAL`, `UV_0`, `TANGENT`, in that
/// order, each tightly packed.
const MESH_VERTEX_STRIDE_BYTES: u64 = 48;

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<RaytracingGeometryId, Blas>,
    compaction_queue: VecDeque<(RaytracingGeometryId, u32, bool)>,
    changed: Vec<RaytracingGeometryId>,
    /// BLAS that are pending deletion, one batch per TLAS build. The back batch collects
    /// retirements since the last build, and every batch ahead of it has one more build to wait
    /// out.
    pending_deletions: VecDeque<Vec<Blas>>,
}

impl BlasManager {
    pub fn get(&self, geometry: &RaytracingGeometryId) -> Option<&Blas> {
        self.blas.get(geometry)
    }

    pub fn device_address(&self, geometry: &RaytracingGeometryId) -> Option<u64> {
        self.blas.get(geometry)?.handle()
    }

    pub fn changed_meshes(&self) -> &[RaytracingGeometryId] {
        &self.changed
    }

    pub fn note_tlas_build(&mut self) {
        if !self.pending_deletions.is_empty() {
            self.pending_deletions.push_back(Vec::new());
        }
    }

    fn insert(&mut self, geometry: RaytracingGeometryId, blas: Blas) {
        if let Some(old) = self.blas.insert(geometry, blas) {
            self.retire(old);
        }

        self.changed.push(geometry);
    }

    fn remove(&mut self, geometry: RaytracingGeometryId) {
        self.changed.push(geometry);

        if let Some(removed) = self.blas.remove(&geometry) {
            self.retire(removed);
        }
    }

    fn retire(&mut self, blas: Blas) {
        match self.pending_deletions.back_mut() {
            Some(batch) => batch.push(blas),
            None => self.pending_deletions.push_back(vec![blas]),
        }
    }
}

/// One BLAS waiting to be built, in the form the build entries need.
struct PendingBlas {
    geometry: RaytracingGeometryId,
    vertex_buffer: Buffer,
    first_vertex: u32,
    vertex_stride: u64,
    index_buffer: Buffer,
    first_index: u32,
    size: BlasTriangleGeometrySizeDescriptor,
}

pub fn prepare_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    extracted_meshes: Res<ExtractedAssets<RenderMesh>>,
    external_meshes: Res<ExtractedExternalMeshes>,
    mesh_allocator: Res<MeshAllocator>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
) {
    blas_manager.changed.clear();

    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.remove(RaytracingGeometryId::Mesh(*asset_id));
    }
    for geometry_id in external_meshes.removed() {
        blas_manager.remove(RaytracingGeometryId::External(*geometry_id));
    }

    let mut pending = Vec::new();

    // Create new BLAS for added or changed bevy meshes
    pending.extend(
        extracted_meshes
            .extracted
            .iter()
            .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh))
            .map(|(asset_id, _)| {
                let vertex_slice = mesh_allocator.mesh_vertex_slice(asset_id).unwrap();
                let index_slice = mesh_allocator.mesh_index_slice(asset_id).unwrap();

                let geometry = RaytracingGeometryId::Mesh(*asset_id);
                let (blas, size) = allocate_blas(
                    vertex_slice.range.len() as u32,
                    index_slice.range.len() as u32,
                    &geometry,
                    &render_device,
                );

                blas_manager.insert(geometry, blas);
                blas_manager
                    .compaction_queue
                    .push_back((geometry, size.vertex_count, false));

                PendingBlas {
                    geometry,
                    vertex_buffer: vertex_slice.buffer.clone(),
                    first_vertex: vertex_slice.range.start,
                    vertex_stride: MESH_VERTEX_STRIDE_BYTES,
                    index_buffer: index_slice.buffer.clone(),
                    first_index: index_slice.range.start,
                    size,
                }
            }),
    );

    // Create new BLAS for added or changed host-built geometry
    pending.extend(
        external_meshes
            .changed()
            .filter(|(_, geometry)| is_external_geometry_compatible(geometry))
            .map(|(id, geometry)| {
                let tracked = RaytracingGeometryId::External(id);
                let (blas, size) = allocate_blas(
                    geometry.vertex_count,
                    geometry.index_count,
                    &tracked,
                    &render_device,
                );

                blas_manager.insert(tracked, blas);
                blas_manager
                    .compaction_queue
                    .push_back((tracked, size.vertex_count, false));

                PendingBlas {
                    geometry: tracked,
                    vertex_buffer: geometry.vertex_buffer.clone(),
                    // The BLAS reads positions from offset 0, which the compatibility check
                    // enforces.
                    first_vertex: 0,
                    vertex_stride: u64::from(geometry.position.byte_stride),
                    index_buffer: geometry.index_buffer.clone(),
                    first_index: 0,
                    size,
                }
            }),
    );

    if pending.is_empty() {
        return;
    }

    // Build geometry into each BLAS
    let build_entries = pending
        .iter()
        .map(|pending| {
            let geometry = BlasTriangleGeometry {
                size: &pending.size,
                vertex_buffer: &pending.vertex_buffer,
                first_vertex: pending.first_vertex,
                vertex_stride: pending.vertex_stride,
                index_buffer: Some(&pending.index_buffer),
                first_index: Some(pending.first_index),
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            BlasBuildEntry {
                blas: &blas_manager.blas[&pending.geometry],
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("blas_build_command_encoder"),
    });
    let time_span = diagnostics
        .as_mut()
        .map(|diagnostics| diagnostics.time_span(&mut command_encoder, "blas_build"));
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    if let Some(time_span) = time_span {
        time_span.end(&mut command_encoder);
    }
    render_queue.submit([command_encoder.finish()]);
}

pub fn compact_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_queue: Res<RenderQueue>,
) {
    let queue_size = blas_manager.compaction_queue.len();
    let mut meshes_processed = 0;
    let mut vertices_compacted = 0;

    while !blas_manager.compaction_queue.is_empty()
        && vertices_compacted < MAX_COMPACTION_VERTICES_PER_FRAME
        && meshes_processed < queue_size
    {
        meshes_processed += 1;

        let (geometry, vertex_count, compaction_started) =
            blas_manager.compaction_queue.pop_front().unwrap();

        let Some(blas) = blas_manager.get(&geometry) else {
            continue;
        };

        if !compaction_started {
            blas.prepare_compaction_async(|_| {});
        }

        if blas.ready_for_compaction() {
            let compacted_blas = render_queue.compact_blas(blas);
            blas_manager.insert(geometry, compacted_blas);

            vertices_compacted += vertex_count;
            continue;
        }

        // BLAS not ready for compaction, put back in queue
        blas_manager
            .compaction_queue
            .push_back((geometry, vertex_count, true));
    }
}

pub fn delete_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_queue: Res<RenderQueue>,
) {
    if blas_manager.pending_deletions.len() <= TLAS_BUILDS_BEFORE_DELETION_ALLOWED {
        return;
    }

    if let Some(deletable) = blas_manager
        .pending_deletions
        .pop_front()
        .filter(|b| !b.is_empty())
    {
        render_queue.on_submitted_work_done(move || drop(deletable));
    }
}

fn allocate_blas(
    vertex_count: u32,
    index_count: u32,
    geometry: &RaytracingGeometryId,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_count),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    // TODO: If we ever introduce BLAS refits, we need to be aware of the TLAS double-buffer
    // to avoid invalidating the previous frame TLAS
    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&geometry.to_string()),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}

fn is_mesh_raytracing_compatible(mesh: &Mesh) -> bool {
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    let vertex_attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .eq([
            (Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_POSITION.format),
            (Mesh::ATTRIBUTE_NORMAL.id, Mesh::ATTRIBUTE_NORMAL.format),
            (Mesh::ATTRIBUTE_UV_0.id, Mesh::ATTRIBUTE_UV_0.format),
            (Mesh::ATTRIBUTE_TANGENT.id, Mesh::ATTRIBUTE_TANGENT.format),
        ]);
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));
    mesh.enable_raytracing && triangle_list && vertex_attributes && indexed_32
}

/// Whether a host-built geometry's declared layout is one the BLAS and the shader can read,
/// logging what is wrong when it is not.
///
/// The BLAS reads positions from offset `0` at `position.byte_stride`, so a layout whose positions
/// start anywhere else cannot be built. Every attribute is read as `f32`, so every offset and
/// stride must be a multiple of `4` bytes. A layout that fails is refused loudly rather than
/// silently, because a refused instance is a black frame either way.
fn is_external_geometry_compatible(geometry: &ExternalMeshGeometry) -> bool {
    let aligned = |attribute: &ExternalVertexAttribute| {
        attribute.byte_offset % 4 == 0 && attribute.byte_stride % 4 == 0
    };
    let layout_ok = geometry.position.byte_offset == 0
        && geometry.position.byte_stride >= 12
        && aligned(&geometry.position)
        && geometry.normal.byte_stride >= 12
        && aligned(&geometry.normal)
        && geometry.uv.is_none_or(|uv| aligned(&uv))
        && geometry.tangent.is_none_or(|tangent| aligned(&tangent));

    if geometry.vertex_count == 0 || geometry.index_count == 0 {
        warn!(
            "bevy_solari: external geometry {} has {} vertices and {} indices",
            geometry.id, geometry.vertex_count, geometry.index_count
        );
        false
    } else if !layout_ok {
        warn!(
            "bevy_solari: external geometry {} has an unusable vertex layout (position offset {}, \
             stride {}; normal offset {}, stride {}). Positions must start at 0 and every offset \
             and stride must be a multiple of 4 bytes",
            geometry.id,
            geometry.position.byte_offset,
            geometry.position.byte_stride,
            geometry.normal.byte_offset,
            geometry.normal.byte_stride,
        );
        false
    } else {
        true
    }
}
