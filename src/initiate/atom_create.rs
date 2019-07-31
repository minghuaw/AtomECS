use crate::maths;
extern crate rand;
use crate::constant;
use crate::constant::PI;
use crate::initiate::*;
use rand::Rng;
extern crate specs;
use crate::atom::*;

use specs::{Component, Entities, Join, LazyUpdate, Read, ReadStorage, System, VecStorage};

pub fn velocity_generate(_t: f64, _mass: f64, _dir: &[f64; 3]) -> [f64; 3] {
	let v_mag = maths::maxwell_generate(_t, _mass);
	let dir = maths::norm(&_dir);
	let dir_1 = maths::norm(&[1.0, 0.0, -dir[0] / dir[2]]);
	let dir_2 = maths::norm(&[
		1.0,
		(dir[1].powf(2.0) - 1.0) / dir[0] / dir[1],
		dir[2] / dir[0],
	]);
	let mut rng = rand::thread_rng();
	let theta = maths::jtheta_gen();
	let theta2 = rng.gen_range(0.0, 2.0 * PI);
	println!("angle one {},angle two {}", theta, theta2);
	let dir_div = maths::array_addition(
		&maths::array_multiply(&dir_1, theta.sin() * theta2.cos()),
		&maths::array_multiply(&dir_2, theta.sin() * theta2.sin()),
	);
	let dirf = maths::array_addition(&maths::array_multiply(&dir, theta.cos()), &dir_div);
	println!("velocity{:?}", maths::array_multiply(&dirf, v_mag));
	assert!(maths::modulus(&dirf) < 1.01 && maths::modulus(&dirf) > 0.99);
	maths::array_multiply(&dirf, v_mag)
}

/// Component representing an oven, which is a source of hot atoms.
pub struct Oven {
	/// Temperature of the oven, in Kelvin
	pub temperature: f64,

	/// Size of the oven's aperture, SI units of metres.
	pub size: [f64; 3],

	/// A vector denoting the direction of the oven.
	pub direction: [f64; 3],

	/// Number of atoms output by the oven every time step
	pub number: u64,
}

impl Component for Oven {
	type Storage = VecStorage<Self>;
}

/// This system creates atoms from an oven source.
///
/// The oven points in the direction [Oven.direction].
pub struct OvenCreateAtomsSystem;

impl<'a> System<'a> for OvenCreateAtomsSystem {
	type SystemData = (
		Entities<'a>,
		ReadStorage<'a, Oven>,
		ReadStorage<'a, AtomInfo>,
		ReadStorage<'a, Position>,
		Read<'a, LazyUpdate>,
	);

	fn run(&mut self, (entities, oven, atom, pos, updater): Self::SystemData) {
		//TODO: Temporary, needs to be fixed properly, eg drawn randomly from distribution. Discuss tomorrow.
		
		let mut rng = rand::thread_rng();

		for (oven, atom, oven_position) in (&oven, &atom, &pos).join() {
			let mass = atom.mass as f64;
			let dir = oven.direction.clone();
			let size = oven.size.clone();
			for _i in 0..oven.number {
				let new_atom = entities.create();
				let new_vel = velocity_generate(oven.temperature, mass*constant::AMU, &dir);
				let pos1 = rng.gen_range(-0.5 * size[0], 0.5 * size[0]);
				let pos2 = rng.gen_range(-0.5 * size[1], 0.5 * size[1]);
				let pos3 = rng.gen_range(-0.5 * size[2], 0.5 * size[2]);
				let start_position = [
					oven_position.pos[0] + pos1,
					oven_position.pos[1] + pos2,
					oven_position.pos[2] + pos3,
				];
				updater.insert(
					new_atom,
					Position {
						pos: start_position,
					},
				);
				updater.insert(new_atom, Velocity { vel: new_vel });
				updater.insert(
					new_atom,
					Force {
						force: [0., 0., 0.],
					},
				);
				updater.insert(new_atom, Mass { value: mass });
				updater.insert(
					new_atom,
					AtomInfo {
						mass:atom.mass,
						mup: atom.mup,
						muz: atom.muz,
						mum: atom.mum,
						frequency: atom.frequency,
						gamma: atom.gamma,
					},
				);
				updater.insert(new_atom,Atom);
				updater.insert(new_atom,NewlyCreated);

				println!("atom created");
			}
		}
	}
}
