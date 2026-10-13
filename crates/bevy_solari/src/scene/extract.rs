use super::{
    ExternalMeshGeometry, RaytracingExternalMesh, RaytracingMesh3d, RaytracingSceneBindings,
};
use bevy_asset::{AssetEvent, AssetId, Assets, Handle};
use bevy_camera::Camera;
use bevy_ecs::{
    entity::Entity,
    lifecycle::RemovedComponents,
    message::MessageReader,
    query::{Added, Changed, Or, With},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_light::{EnvironmentMapLight, GeneratedEnvironmentMapLight};
use bevy_math::Quat;
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;
use bevy_utils::once;
use tracing::warn;

/// Creates or removes components in the render world related to raytracing instances.
pub fn extract_raytracing_scene_structural(
    new_instances: Extract<
        Query<
            (
                RenderEntity,
                Option<&RaytracingMesh3d>,
                Option<&RaytracingExternalMesh>,
                &MeshMaterial3d<StandardMaterial>,
                &GlobalTransform,
                Option<&PreviousGlobalTransform>,
            ),
            Or<(Added<RaytracingMesh3d>, Added<RaytracingExternalMesh>)>,
        >,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    mut removed_external_meshes: Extract<RemovedComponents<RaytracingExternalMesh>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    // Process removed components before additions, that way it properly handles same-frame removal->insertion
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }
    for main_entity in removed_external_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands
                .entity(render_entity)
                .remove::<RaytracingExternalMesh>();
        }
    }

    for (render_entity, mesh, external, material, transform, previous_frame_transform) in
        &new_instances
    {
        let mut entity = commands.entity(render_entity);
        entity.insert((
            material.clone(),
            *transform,
            previous_frame_transform
                .cloned()
                .unwrap_or(PreviousGlobalTransform(transform.affine())),
        ));
        if let Some(mesh) = mesh {
            entity.insert(mesh.clone());
        }
        if let Some(external) = external {
            entity.insert(external.clone());
        }
    }
}

/// Copies the transforms of moved raytracing instances from the main world
/// straight into their GPU buffers.
pub fn extract_raytracing_scene_transforms(
    main_instances: Extract<
        Query<
            (
                RenderEntity,
                &GlobalTransform,
                Option<&PreviousGlobalTransform>,
            ),
            (
                Or<(Changed<GlobalTransform>, Changed<PreviousGlobalTransform>)>,
                Or<(With<RaytracingMesh3d>, With<RaytracingExternalMesh>)>,
            ),
        >,
    >,
    bindings: Res<RaytracingSceneBindings>,
) {
    main_instances
        .par_iter()
        .for_each(|(render_entity, transform, previous_frame_transform)| {
            let previous_frame_transform = previous_frame_transform
                .cloned()
                .unwrap_or(PreviousGlobalTransform(transform.affine()));

            bindings.move_instance(render_entity, transform, &previous_frame_transform);
        });
}

/// Updates the mesh and material of existing raytracing instances in the render world.
pub fn extract_raytracing_scene_meshes_and_materials(
    main_instances: Extract<
        Query<
            (
                RenderEntity,
                Option<&RaytracingMesh3d>,
                Option<&RaytracingExternalMesh>,
                &MeshMaterial3d<StandardMaterial>,
            ),
            Or<(
                Changed<RaytracingMesh3d>,
                Changed<RaytracingExternalMesh>,
                Changed<MeshMaterial3d<StandardMaterial>>,
            )>,
        >,
    >,
    mut render_instances: Query<(
        &mut MeshMaterial3d<StandardMaterial>,
        Option<&mut RaytracingMesh3d>,
        Option<&mut RaytracingExternalMesh>,
    )>,
) {
    for (render_entity, new_mesh, new_external, new_material) in &main_instances {
        if let Ok((mut material, mesh, external)) = render_instances.get_mut(render_entity) {
            *material = new_material.clone();
            if let (Some(new), Some(mut current)) = (new_mesh, mesh) {
                *current = new.clone();
            }
            if let (Some(new), Some(mut current)) = (new_external, external) {
                *current = new.clone();
            }
        }
    }
}

/// External geometry mirrored into the render world, by id.
///
/// `changed` drives BLAS builds, `removed` drives BLAS deletions, and `live` resolves an
/// instance's geometry while it exists. The reference count keeps a shared geometry's BLAS alive
/// until the last placement naming it is gone.
#[derive(Resource, Default)]
pub struct ExtractedExternalMeshes {
    changed: Vec<(u64, ExternalMeshGeometry)>,
    removed: Vec<u64>,
    live: HashMap<u64, u32>,
    entities: HashMap<Entity, u64>,
}

impl ExtractedExternalMeshes {
    /// Every geometry added or changed this frame.
    pub fn changed(&self) -> impl Iterator<Item = (u64, &ExternalMeshGeometry)> {
        self.changed.iter().map(|(id, geometry)| (*id, geometry))
    }

    /// Geometry ids whose last placement went away this frame.
    pub fn removed(&self) -> &[u64] {
        &self.removed
    }

    fn acquire(&mut self, entity: Entity, geometry: &ExternalMeshGeometry) {
        if self.entities.get(&entity) == Some(&geometry.id) {
            if self.live.contains_key(&geometry.id) {
                self.changed.push((geometry.id, geometry.clone()));
                return;
            }
        } else if let Some(previous) = self.entities.insert(entity, geometry.id) {
            self.release(previous);
        }

        *self.live.entry(geometry.id).or_insert(0) += 1;
        self.changed.push((geometry.id, geometry.clone()));
    }

    fn release(&mut self, id: u64) {
        let empty = self.live.get_mut(&id).is_some_and(|count| {
            *count -= 1;
            *count == 0
        });
        if empty {
            self.live.remove(&id);
            self.removed.push(id);
        }
    }
}

/// Keeps [`ExtractedExternalMeshes`] in step with the host-built geometry in the scene.
pub fn extract_raytracing_external_meshes(
    mut extracted: ResMut<ExtractedExternalMeshes>,
    changed: Extract<Query<(Entity, &RaytracingExternalMesh), Changed<RaytracingExternalMesh>>>,
    mut removed: Extract<RemovedComponents<RaytracingExternalMesh>>,
) {
    extracted.changed.clear();
    extracted.removed.clear();

    for (entity, external) in &changed {
        extracted.acquire(entity, &external.0);
    }

    for main_entity in removed.read() {
        if let Some(id) = extracted.entities.remove(&main_entity) {
            extracted.release(id);
        }
    }
}

/// The set of [`StandardMaterial`] in the scene, mirrored into the render world.
#[derive(Resource, Default)]
pub struct StandardMaterialAssets {
    materials: HashMap<AssetId<StandardMaterial>, StandardMaterial>,
    /// Materials added or modified this frame.
    pub changed: Vec<AssetId<StandardMaterial>>,
    /// Materials removed this frame.
    pub removed: Vec<AssetId<StandardMaterial>>,
}

impl StandardMaterialAssets {
    pub fn get(&self, id: &AssetId<StandardMaterial>) -> Option<&StandardMaterial> {
        self.materials.get(id)
    }
}

/// Keeps [`StandardMaterialAssets`] up to date in the render world.
pub fn extract_raytracing_material_assets(
    main_materials: Extract<Res<Assets<StandardMaterial>>>,
    mut render_materials: ResMut<StandardMaterialAssets>,
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
) {
    let render_materials = &mut *render_materials;

    render_materials.changed.clear();
    render_materials.removed.clear();

    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if let Some(material) = main_materials.get(*id) {
                    render_materials.materials.insert(*id, material.clone());
                    render_materials.changed.push(*id);
                }
            }
            AssetEvent::Removed { id } => {
                render_materials.materials.remove(id);
                render_materials.removed.push(*id);
            }
            AssetEvent::Unused { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
}

#[derive(Resource, Default, Clone, PartialEq)]
pub struct ExtractedEnvironmentMapLight {
    pub cubemap: Option<Handle<Image>>,
    pub intensity: f32,
    pub rotation: Quat,
}

/// Finds the environment map light to use for the raytraced scene, if any.
pub fn extract_raytracing_environment_map_light(
    cameras: Extract<
        Query<(
            &Camera,
            Option<&GeneratedEnvironmentMapLight>,
            Option<&EnvironmentMapLight>,
        )>,
    >,
    mut environment_map_light: ResMut<ExtractedEnvironmentMapLight>,
) {
    let mut extracted_env_map_light = ExtractedEnvironmentMapLight::default();

    for (camera, generated, pregenerated) in &cameras {
        if !camera.is_active {
            continue;
        }

        let env_map_light = match (generated, pregenerated) {
            (Some(generated), _) => ExtractedEnvironmentMapLight {
                cubemap: Some(generated.environment_map.clone()),
                intensity: generated.intensity,
                rotation: generated.rotation,
            },
            (None, Some(pregenerated)) => ExtractedEnvironmentMapLight {
                cubemap: Some(pregenerated.specular_map.clone()),
                intensity: pregenerated.intensity,
                rotation: pregenerated.rotation,
            },
            (None, None) => continue,
        };

        if extracted_env_map_light.cubemap.is_none() {
            extracted_env_map_light = env_map_light;
        } else if extracted_env_map_light != env_map_light {
            once!(warn!(
                "bevy_solari only supports a single environment light for the whole scene, but \
                 multiple cameras have differing environment lights. Using the first one found."
            ));
        }
    }

    *environment_map_light = extracted_env_map_light;
}
