extern crate nalgebra;
extern crate specs;
use crate::initiate::NewlyCreated;
use nalgebra::Vector3;
use specs::{
	Component, DispatcherBuilder, Entities, Join, LazyUpdate, Read, ReadStorage, System,
	VecStorage, World, WriteStorage,
};

pub mod grid;
pub mod quadrupole;
pub mod uniform;

/// A component that stores the magnetic field at an entity's location.
#[derive(Copy, Clone)]
pub struct MagneticFieldSampler {
	/// Vector representing the magnetic field components along x,y,z in units of Tesla.
	pub field: Vector3<f64>,

	/// Magnitude of the magnetic field in units of Tesla
	pub magnitude: f64,
}
impl MagneticFieldSampler {
	pub fn tesla(b_field: Vector3<f64>) -> Self {
		MagneticFieldSampler {
			field: b_field,
			magnitude: b_field.norm(),
		}
	}
}
impl Component for MagneticFieldSampler {
	type Storage = VecStorage<Self>;
}

impl Default for MagneticFieldSampler {
	fn default() -> Self {
		MagneticFieldSampler {
			field: Vector3::new(0.0, 0.0, 0.0),
			magnitude: 0.0,
		}
	}
}

/// System that clears the magnetic field samplers each frame.
pub struct ClearMagneticFieldSamplerSystem;

impl<'a> System<'a> for ClearMagneticFieldSamplerSystem {
	type SystemData = WriteStorage<'a, MagneticFieldSampler>;
	fn run(&mut self, mut sampler: Self::SystemData) {
		for sampler in (&mut sampler).join() {
			sampler.magnitude = 0.;
			sampler.field = Vector3::new(0.0, 0.0, 0.0)
		}
	}
}

/// System that calculates the magnitude of the magnetic field.
///
/// The magnetic field magnitude is frequently used, so it makes sense to calculate it once and cache the result.
/// This system runs after all other magnetic field systems.
pub struct CalculateMagneticFieldMagnitudeSystem;

impl<'a> System<'a> for CalculateMagneticFieldMagnitudeSystem {
	type SystemData = WriteStorage<'a, MagneticFieldSampler>;
	fn run(&mut self, mut sampler: Self::SystemData) {
		for sampler in (&mut sampler).join() {
			sampler.magnitude = sampler.field.norm();
		}
	}
}

/// Attachs the MagneticFieldSampler component to newly created atoms.
/// This allows other magnetic Systems to interact with the atom, eg to calculate fields at their location.
pub struct AttachFieldSamplersToNewlyCreatedAtomsSystem;

impl<'a> System<'a> for AttachFieldSamplersToNewlyCreatedAtomsSystem {
	type SystemData = (
		Entities<'a>,
		ReadStorage<'a, NewlyCreated>,
		Read<'a, LazyUpdate>,
	);
	fn run(&mut self, (ent, newly_created, updater): Self::SystemData) {
		for (ent, _nc) in (&ent, &newly_created).join() {
			updater.insert(ent, MagneticFieldSampler::default());
		}
	}
}

/// Adds the systems required by magnetics to the dispatcher.
///
/// #Arguments
///
/// `builder`: the dispatch builder to modify
///
/// `deps`: any dependencies that must be completed before the magnetics systems run.
pub fn add_systems_to_dispatch(
	builder: DispatcherBuilder<'static, 'static>,
	deps: &[&str],
) -> DispatcherBuilder<'static, 'static> {
	builder
		.with(ClearMagneticFieldSamplerSystem, "magnetics_clear", deps)
		.with(
			quadrupole::Sample3DQuadrupoleFieldSystem,
			"magnetics_quadrupole",
			&["magnetics_clear"],
		)
		.with(
			uniform::UniformMagneticFieldSystem,
			"magnetics_uniform",
			&["magnetics_quadrupole"],
		)
		.with(
			grid::SampleMagneticGridSystem,
			"magnetics_grid",
			&["magnetics_uniform"],
		)
		.with(
			CalculateMagneticFieldMagnitudeSystem,
			"magnetics_magnitude",
			&["magnetics_grid"],
		)
		.with(
			AttachFieldSamplersToNewlyCreatedAtomsSystem,
			"add_magnetic_field_samplers",
			&[],
		)
}

/// Registers resources required by magnetics to the ecs world.
pub fn register_components(world: &mut World) {
	world.register::<uniform::UniformMagneticField>();
	world.register::<quadrupole::QuadrupoleField3D>();
	world.register::<MagneticFieldSampler>();
	world.register::<grid::PrecalculatedMagneticFieldGrid>();
}

#[cfg(test)]
pub mod tests {

	use super::*;
	extern crate specs;
	use crate::atom::Position;
	use specs::{Builder, DispatcherBuilder, World};

	/// Tests the correct implementation of the magnetics systems and dispatcher.
	/// This is done by setting up a test world and ensuring that the magnetic systems perform the correct operations on test entities.
	#[test]
	fn test_magnetics_systems() {
		let mut test_world = World::new();
		register_components(&mut test_world);
		test_world.register::<Position>();
		let builder = DispatcherBuilder::new();
		let configured_builder = add_systems_to_dispatch(builder, &[]);
		let mut dispatcher = configured_builder.build();
		dispatcher.setup(&mut test_world.res);

		test_world
			.create_entity()
			.with(uniform::UniformMagneticField {
				field: Vector3::new(2.0, 0.0, 0.0),
			})
			.with(quadrupole::QuadrupoleField3D::gauss_per_cm(
				100.0,
				Vector3::z(),
			))
			.with(Position {
				pos: Vector3::new(0.0, 0.0, 0.0),
			})
			.build();

		let sampler_entity = test_world
			.create_entity()
			.with(Position {
				pos: Vector3::new(1.0, 1.0, 1.0),
			})
			.with(MagneticFieldSampler::default())
			.build();

		dispatcher.dispatch(&mut test_world.res);

		let samplers = test_world.read_storage::<MagneticFieldSampler>();
		let sampler = samplers.get(sampler_entity);
		assert_eq!(
			sampler.expect("entity not found").field,
			Vector3::new(2.0 + 1.0, 1.0, -2.0)
		);
	}

	/// Tests that magnetic field samplers are added to newly created atoms.
	#[test]
	fn test_field_samplers_are_added() {
		let mut test_world = World::new();
		register_components(&mut test_world);
		test_world.register::<NewlyCreated>();
		let builder = DispatcherBuilder::new();
		let configured_builder = add_systems_to_dispatch(builder, &[]);
		let mut dispatcher = configured_builder.build();
		dispatcher.setup(&mut test_world.res);

		let sampler_entity = test_world.create_entity().with(NewlyCreated).build();

		dispatcher.dispatch(&mut test_world.res);
		test_world.maintain();

		let samplers = test_world.read_storage::<MagneticFieldSampler>();
		assert_eq!(samplers.contains(sampler_entity), true);
	}
}
