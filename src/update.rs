extern crate specs;
use specs::{System,ReadStorage,WriteStorage,Join,ReadExpect};

use crate::atom::*;
use crate::laser::InteractionLaserALL;
use crate::maths;
use crate::integrator::*;
use crate::initiate::*;
use crate::constant;
extern crate rand;
use rand::Rng;

pub struct UpdateForce;

impl <'a>System<'a> for UpdateForce{
	// this system will update the force component for atoms based on interaction with lasers and random kick
	type SystemData = ( WriteStorage<'a,Force>,
									ReadStorage<'a,Gravity>,
									ReadStorage<'a,InteractionLaserALL>,
									ReadStorage<'a,RandKick>
									);
									
	fn run(&mut self,(mut force,gravity,inter,kick):Self::SystemData){
		for (mut force, inter) in (&mut force,&inter).join(){
			let mut new_force = [0.,0.,0.];
			//println!("force updated");
		
			for interall in inter.content.iter(){
				new_force = maths::array_addition(&new_force,&interall.force);
			}
			force.force = new_force;
		}
		for (mut force,kick) in (&mut force,&kick).join(){
			force.force = maths::array_addition(&kick.force,&force.force);
		}
		for (mut force,gravity) in (&mut force,&gravity).join(){
			force.force = maths::array_addition(&force.force,&gravity.force);
		}
	}
}

pub struct UpdateRandKick;
//this system must be ran after update_force
impl <'a>System<'a> for UpdateRandKick{
	type SystemData = (ReadStorage<'a,InteractionLaserALL>,
								WriteStorage<'a,RandKick>,
								ReadExpect<'a,Timestep>,
								ReadStorage<'a,AtomInfo>);	
	fn run(&mut self, (interall,mut kick,t,atom):Self::SystemData){
		// to the best of the knowledge, the number of actual random kick should be calculated using a possoin distribution
		for (interall,mut kick,atom) in (&interall,&mut kick,&atom).join(){
			//this system will look at forces due to interaction with all the lasers and calculate the corresponding number of random kick involved
			let mut total_impulse = 0.0 ; 
			kick.force =[0.,0.,0.];
			for interaction in &interall.content{
				total_impulse = total_impulse + maths::modulus(&interaction.force)*t.delta;
			}
			let momentum_photon = constant::HBAR * 2.*constant::PI*atom.frequency/constant::C;
			let mut num_kick = total_impulse/ momentum_photon;
			//num_kick will be the expected number of random kick involved
			loop{
				if num_kick >1.{
					// if the number is bigger than 1, a random kick will be added with direction random
					num_kick = num_kick - 1.;
					kick.force = maths::array_addition(&kick.force,&maths::array_multiply(&maths::random_direction(),momentum_photon/t.delta));
				}
				else{
					// if the remaining kick is smaller than 0, there is a chance that the kick is random
					let mut rng = rand::thread_rng();
					let result = rng.gen_range(0.0, 1.0);
					if result < num_kick{
						kick.force = maths::array_addition(&kick.force,&maths::array_multiply(&maths::random_direction(),momentum_photon/t.delta));
					}
					break;
				}
			}
		}
	}
}

#[cfg(test)]
pub mod tests {

	use super::*;
	extern crate specs;
	use crate::laser::InteractionLaser;

	/// Tests the correct implementation of update force
	#[test]
	
	/// Tests the correct implementation of the magnetics systems and dispatcher.
	/// This is done by setting up a test world and ensuring that the magnetic systems perform the correct operations on test entities.
	#[test]
	fn test_update_systems()
	{
		use specs::{RunNow,World,Builder};
		let mut test_world = World::new();
		test_world.register::<RandKick>();
		test_world.register::<Force>();
		test_world.register::<InteractionLaserALL>();
		test_world.register::<Gravity>();

		let mut content = Vec::new();
		content.push(InteractionLaser{wavenumber:[1.,1.,2.],index:1,intensity:1.,polarization:1.,detuning_doppler:1.,force:[1.,0.,0.]});
		content.push(InteractionLaser{wavenumber:[1.,1.,2.],index:2,intensity:1.,polarization:1.,detuning_doppler:1.,force:[2.,0.,0.]});

		let test_interaction = InteractionLaserALL{content};
		let test_kick = RandKick{force:[1.,0.,0.]};
		let sample_entity= test_world.create_entity().
		with(test_interaction).
		with(test_kick).
		with(Force{force:[0.,0.,0.]}).build();

		let mut update_test = UpdateForce;
		update_test.run_now(&test_world.res);



		let samplers = test_world.read_storage::<Force>();
		let sampler = samplers.get(sample_entity);
		assert_eq!(sampler.expect("entity not found").force,[4.,0.,0.]);
	}
}	