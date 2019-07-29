extern crate specs;
use specs::{System,ReadStorage,WriteStorage,Join,Component,VecStorage};
use crate::atom::Position;
use crate::maths::Maths;

/// A component that measures the magnetic field at a point in space.
pub struct MagneticFieldSampler{
	
	/// Vector representing the magnetic field components along x,y,z in units of Gauss.
	pub field:[f64;3],

	/// Magnitude of the magnetic field in units of Gauss
	pub magnitude: f64
}

impl Component for MagneticFieldSampler{
	type Storage = VecStorage<Self>;
}

/// A component representing a 3D quadrupole field.
pub struct QuadrupoleField3D{
	/// Gradient of the quadrupole field, in units of Gauss/cm
	pub gradient:f64
}

impl Component for QuadrupoleField3D{
	type Storage = VecStorage<Self>;
}

/// Updates the values of magnetic field samplers to include quadrupole fields in the world.
pub struct Sample3DQuadrupoleFieldSystem;

impl <'a> System<'a> for Sample3DQuadrupoleFieldSystem{
		type SystemData = (WriteStorage<'a,MagneticFieldSampler>,
									ReadStorage<'a,Position>,
									ReadStorage<'a,QuadrupoleField3D>,
									);
	fn run(&mut self,(mut _sampler,pos,_quadrupoles):Self::SystemData){
		
		for (centre, quadrupole) in (&pos, &_quadrupoles).join(){
			for (pos,mut sampler) in (&pos,&mut _sampler).join(){
				let rela_pos = Maths::array_addition(&pos.pos,&Maths::array_multiply(&centre.pos,-1.));
				sampler.field = Maths::array_multiply(&[-rela_pos[0],-rela_pos[1],2.0*rela_pos[2]],quadrupole.gradient);

				//TODO: Set value for sampler.magnitude
				// sampler.magnitude = 

				//TODO: Value should be added to magnetic field sampler, which is cleared at the start of every frame. This will then support multiple different field sources.
			}
		}
	}
}