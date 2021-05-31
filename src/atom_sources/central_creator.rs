//! Introduces a component `CentralCreator` for the source entity which spawns atoms with
//! desired DensityDistribution and VelocityDistribution
//!
//! The main use of this feature is to create a starting point for a second MOT stage where the atoms are
//! not generated by an oven source. For example if one only seeks to simulate a red Sr MOT, one
//! sets the distributions in the central creator such that they resemble the final conditions
//! in the first MOT stage.
//!
//! This is accomplished by 5 different types of distributions, 3 scalar and 2 vector.

extern crate nalgebra;
use nalgebra::Vector3;
use rand::Rng;
use specs::prelude::*;

use super::emit::AtomNumberToEmit;
use super::mass::MassDistribution;
use super::VelocityCap;
use crate::atom::*;
use crate::initiate::*;

// Define some distributions that are necessary to custom-create the initial
// conditions of the atoms created

/// Represents the probability distribution to find an atom at
/// a specific position in space.
///
/// Sample from this to get a position in space (could be a sphere or disc...)
///
/// More complicated distribution types will likely be introduced in the future
#[derive(Copy, Clone)]
pub enum PositionDensityDistribution {
    UniformCuboidic { size: [f64; 3] },
    UniformSpheric { radius: f64 },
}

///  Distribution of the characteristic velocity of a particle at a certain position in space
///
/// This is just existing to keep it general in case we would like to represent
/// that particles in the middle are faster etc.
#[derive(Copy, Clone)]
pub enum SpatialSpeedDistribution {
    Uniform { speed: f64 },
    UniformCuboidic { speed: f64, size: [f64; 3] },
    UniformSpheric { speed: f64, radius: f64 },
}

/// Speed distribution depending on the characteristic speed
///
/// Depending on your characteristic speed value,  a speed distribution is produced,
///  sample from that to get the actual speed.
#[derive(Copy, Clone)]
pub enum SpeedDensityDistribution {
    // for UniformCentral: distribution like ____-----____ where width is width of -----
    // and the characteristic speed is center of support
    UniformCentral { width: f64 },
}

/// Vectorfield as function of position in space
///
/// Depending on your position, get a characteristic vector (for example pointing inwards)
#[derive(Copy, Clone)]
pub enum SpatialVectorDistribution {
    /// no preferred direction, everywhere
    Uniform {},
}

/// Probability function in (phi, theta) of getting a certain vector. Takes a characteristic vector.
///
///  Depending on your characteristic vector, create a vector distribution.
/// Sample from that to get the direction your velocity is pointing in.
#[derive(Copy, Clone)]
pub enum VectorDensityDistribution {
    /// all directions equally probable. Practically ignores the characteristic vector.
    Uniform {},
}

/// CentralCreator is a component that can be attached to a source entity.
///
/// It spawns atoms based on position in a coordinate system (which normally has the
/// same center as the entire setup), hence the name
///
/// It is designed in analogy to the Oven but without a builder (yet, might come later)
pub struct CentralCreator {
    position_density_distribution: PositionDensityDistribution,
    spatial_speed_distribution: SpatialSpeedDistribution,
    speed_density_distribution: SpeedDensityDistribution,
    spatial_vector_distribution: SpatialVectorDistribution,
    vector_density_distribution: VectorDensityDistribution,
}

impl CentralCreator {
    // Create a new trivial, cubic central creator
    pub fn new_uniform_cubic(size_of_cube: f64, speed: f64) -> Self {
        Self {
            position_density_distribution: PositionDensityDistribution::UniformCuboidic {
                size: [size_of_cube, size_of_cube, size_of_cube],
            },
            spatial_speed_distribution: SpatialSpeedDistribution::Uniform { speed: speed },
            speed_density_distribution: SpeedDensityDistribution::UniformCentral {
                width: 0.5 * speed,
            },
            spatial_vector_distribution: SpatialVectorDistribution::Uniform {},
            vector_density_distribution: VectorDensityDistribution::Uniform {},
        }
    }

    // sample frome the central_creator and get random position and velocity vectors
    pub fn get_random_spawn_condition(&self) -> (Vector3<f64>, Vector3<f64>) {
        let mut rng = rand::thread_rng();

        let pos_vector = match self.position_density_distribution {
            PositionDensityDistribution::UniformCuboidic { size } => {
                let size = size.clone();
                let pos1 = rng.gen_range(-0.5 * size[0], 0.5 * size[0]);
                let pos2 = rng.gen_range(-0.5 * size[1], 0.5 * size[1]);
                let pos3 = rng.gen_range(-0.5 * size[2], 0.5 * size[2]);
                nalgebra::Vector3::new(pos1, pos2, pos3)
            }
            PositionDensityDistribution::UniformSpheric { radius: _ } => {
                // Not implemented!
                panic!("get_random_spawn_condition for PositionDensityDistribution::UniformSpheric not yet implemented!");
            }
        };

        let characteristic_speed: f64 = match self.spatial_speed_distribution {
            SpatialSpeedDistribution::Uniform { speed } => speed,
            SpatialSpeedDistribution::UniformCuboidic { speed: _, size: _ } => {
                // Not implemented!
                panic!("get_random_spawn_condition for SpatialSpeedDistribution::UniformCuboidic not yet implemented!");
            }
            SpatialSpeedDistribution::UniformSpheric {
                speed: _,
                radius: _,
            } => {
                // Not implemented!
                panic!("get_random_spawn_condition for SpatialSpeedDistribution::UniformSpheric not yet implemented!");
            }
        };

        let speed: f64 = match self.speed_density_distribution {
            SpeedDensityDistribution::UniformCentral { width } => {
                let min: f64 = (0.0f64).max(characteristic_speed - width);
                rng.gen_range(min, characteristic_speed + width)
            }
        };

        // so far this is ignored by the VectorDensityDistribution::Uniform {}
        // but this changes for mor complex VectorDensityDistributions
        let _characteristic_vector: Vector3<f64> = match self.spatial_vector_distribution {
            SpatialVectorDistribution::Uniform {} => Vector3::new(0.0, 0.0, 0.0),
        };

        let vector: Vector3<f64> = match self.vector_density_distribution {
            VectorDensityDistribution::Uniform {} => {
                let vec1 = rng.gen_range(-1.0, 1.0);
                let vec2 = rng.gen_range(-1.0, 1.0);
                let vec3 = rng.gen_range(-1.0, 1.0);
                (nalgebra::Vector3::new(vec1, vec2, vec3)).normalize()
            }
        };

        (pos_vector, speed * vector)
    }
}

impl Component for CentralCreator {
    type Storage = HashMapStorage<Self>;
}

/// Creates atoms from an `CentralCreator` source.
pub struct CentralCreatorCreateAtomsSystem;

impl<'a> System<'a> for CentralCreatorCreateAtomsSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, CentralCreator>,
        ReadStorage<'a, AtomicTransition>,
        ReadStorage<'a, AtomNumberToEmit>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, MassDistribution>,
        Option<Read<'a, VelocityCap>>,
        Read<'a, LazyUpdate>,
    );
    fn run(
        &mut self,
        (
            entities,
            central_creator,
            atom,
            numbers_to_emit,
            pos,
            mass_distribution,
            velocity_cap,
            updater,
        ): Self::SystemData,
    ) {
        let max_vel = match velocity_cap {
            Some(cap) => cap.value,
            None => std::f64::MAX,
        };

        for (central_creator, atom, number_to_emit, _creator_position, mass_distribution) in (
            &central_creator,
            &atom,
            &numbers_to_emit,
            &pos,
            &mass_distribution,
        )
            .join()
        {
            for _i in 0..number_to_emit.number {
                let mass = mass_distribution.draw_random_mass();
                let (start_position, start_velocity) = central_creator.get_random_spawn_condition();

                if start_velocity.norm() > max_vel {
                    continue;
                }

                let new_atom = entities.create();
                updater.insert(
                    new_atom,
                    Position {
                        pos: start_position,
                    },
                );
                updater.insert(
                    new_atom,
                    Velocity {
                        vel: start_velocity.clone(),
                    },
                );
                updater.insert(new_atom, Force::new());
                updater.insert(new_atom, mass);
                updater.insert(new_atom, atom.clone());
                updater.insert(new_atom, Atom);
                updater.insert(
                    new_atom,
                    InitialVelocity {
                        vel: start_velocity,
                    },
                );
                updater.insert(new_atom, NewlyCreated);
            }
        }
    }
}

#[cfg(test)]
pub mod tests {

    use super::*;
    extern crate nalgebra;
    use nalgebra::Vector3;

    /// Tests the correct implementation of the `SampleLaserIntensitySystem`
    #[test]
    fn test_central_creator_create_atoms_system() {
        let mut test_world = World::new();

        test_world.register::<CentralCreator>();
        test_world.register::<AtomicTransition>();
        test_world.register::<AtomNumberToEmit>();
        test_world.register::<Position>();
        test_world.register::<Velocity>();
        test_world.register::<InitialVelocity>();
        test_world.register::<Force>();
        test_world.register::<Atom>();
        test_world.register::<MassDistribution>();
        test_world.register::<NewlyCreated>();
        test_world.register::<CheckerComponent>();
        test_world.register::<Mass>();

        let number_to_emit = 100;
        let size_of_cube = 1.0; //m
        let speed = 1.0; // m/s

        test_world
            .create_entity()
            .with(CentralCreator::new_uniform_cubic(size_of_cube, speed))
            .with(Position {
                pos: Vector3::new(0.0, 0.0, 0.0),
            })
            .with(MassDistribution::new(vec![
                crate::atom_sources::mass::MassRatio {
                    mass: 88.0,
                    ratio: 1.0,
                },
            ]))
            .with(AtomicTransition::strontium_red())
            .with(AtomNumberToEmit {
                number: number_to_emit,
            })
            .build();

        let mut system = CentralCreatorCreateAtomsSystem;
        system.run_now(&test_world);
        test_world.maintain();
        pub struct CheckerComponent {
            everything_allright: bool,
            number: i32,
        }

        impl Component for CheckerComponent {
            type Storage = specs::VecStorage<Self>;
        }

        let checker1 = test_world
            .create_entity()
            .with(CheckerComponent {
                everything_allright: true,
                number: 0,
            })
            .build();

        pub struct TestSystem;

        impl<'a> System<'a> for TestSystem {
            type SystemData = (
                specs::WriteStorage<'a, CheckerComponent>,
                ReadStorage<'a, Position>,
                ReadStorage<'a, Velocity>,
                ReadStorage<'a, Atom>,
            );
            fn run(&mut self, (mut checker, position, velocity, atom): Self::SystemData) {
                for mut check in (&mut checker).join() {
                    for (pos, vel, _atom) in (&position, &velocity, &atom).join() {
                        if pos.pos.norm() <= 3.0_f64.powf(0.5) * 1.0 {
                        } else {
                            check.everything_allright = false;
                        }
                        if vel.vel.norm() <= 3.0_f64.powf(0.5) * 1.0 {
                        } else {
                            check.everything_allright = false;
                        }
                        check.number = check.number + 1;
                    }
                }
            }
        }
        TestSystem.run_now(&test_world);
        test_world.maintain();

        let sampler_storage = test_world.read_storage::<CheckerComponent>();

        assert!(
            sampler_storage
                .get(checker1)
                .expect("entity not found")
                .everything_allright,
            true
        );

        assert_eq!(
            sampler_storage
                .get(checker1)
                .expect("entity not found")
                .number,
            number_to_emit
        );
    }
}
