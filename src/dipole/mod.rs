//! A module that implements systems and components for dipole trapping in AtomECS.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use specs::DispatcherBuilder;

use crate::laser::LaserPlugin;
use crate::{constant, simulation::Plugin};
use crate::laser::index::LaserIndex;

use serde::{Deserialize, Serialize};
use specs::prelude::*;

use ordered_float::OrderedFloat;

pub mod force;

/// A component marking the entity as laser beam for dipole forces and
/// holding properties of the light
#[derive(Deserialize, Serialize, Clone, Copy)]
pub struct DipoleLight {
    ///wavelength of the laser light in SI units of m.
    pub wavelength: f64,
}
impl Component for DipoleLight {
    type Storage = HashMapStorage<Self>;
}

impl DipoleLight {
    /// Frequency of the dipole light in units of Hz
    pub fn frequency(&self) -> f64 {
        constant::C / self.wavelength
    }

    /// Wavenumber of the dipole light, in units of 2pi/m
    pub fn wavenumber(&self) -> f64 {
        2.0 * constant::PI / self.wavelength
    }
}

/// A custom type that simply wraps a `f64` representing the wavelength of a laser beam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wavelength {
    value: OrderedFloat<f64>,
}

impl Wavelength {
    pub fn new(value: f64) -> Self {
        Wavelength {
            value: OrderedFloat(value),
        }
    }
}

/// A component that represents the optical transition wavelength of an atom.
/// 
/// Added by Minghua Wu on Sep. 22, 2023
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionWavelength {
    value: OrderedFloat<f64>,
}
impl Component for TransitionWavelength {
    type Storage = VecStorage<Self>;
}

impl TransitionWavelength {
    pub fn new(value: f64) -> Self {
        TransitionWavelength {
            value: OrderedFloat(value),
        }
    }
}

/// A component that represents the optical transition linewidth of an atom.
/// 
/// Added by Minghua Wu on Sep. 22, 2023
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionLinewidth {
    value: OrderedFloat<f64>,
}
impl Component for TransitionLinewidth {
    type Storage = VecStorage<Self>;
}

impl TransitionLinewidth {
    pub fn new(value: f64) -> Self {
        TransitionLinewidth {
            value: OrderedFloat(value),
        }
    }
}

/// A wrapper around `Wavelength`, `OpticalTransitionWavelength` and 
/// `OpticalTransitionLinewidth` that can be used as a key in a `HashMap`. 
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolarizabilityMapKey {
    pub wavelength: Wavelength,
    pub transition_wavelength: TransitionWavelength,
    pub transition_linewidth: TransitionLinewidth,
}

impl PolarizabilityMapKey {
    pub fn new(
        wavelength: f64,
        transition_wavelength: f64,
        transition_linewidth: f64,
    ) -> Self {
        PolarizabilityMapKey {
            wavelength: Wavelength::new(wavelength),
            transition_wavelength: TransitionWavelength::new(transition_wavelength),
            transition_linewidth: TransitionLinewidth::new(transition_linewidth),
        }
    }
}

/// A resource that maps PolarizabilityMapKey to Polarizability.
/// 
/// Added by Minghua Wu on Sep. 22, 2023
pub struct PolarizabilityMap(pub HashMap<PolarizabilityMapKey, Polarizability>);

impl PolarizabilityMap {
    pub fn new() -> Self {
        PolarizabilityMap(HashMap::new())
    }
}

impl Deref for PolarizabilityMap {
    type Target = HashMap<PolarizabilityMapKey, Polarizability>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PolarizabilityMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An atom component that represents the polarizability of the atom in a `DipoleLight` laser beam.
///
/// The force exterted on the atom is equal to:
/// `force = polarizability.prefactor * intensity_gradient`
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct Polarizability {
    /// The prefactor is a constant of proportionality that relates the intensity gradient (in W/m) to the force on the atom (in N).
    pub prefactor: f64,
}

// // This is commented out to make sure that the user does not accidentally add this component to an atom.
// impl Component for Polarizability {
//     type Storage = VecStorage<Self>;
// }

impl Polarizability {
    /// Calculate the polarizability of an atom in a dipole beam of given wavelength, detuned from a strong optical transition.
    ///
    /// The wavelengths of both transitions are in SI units of m.
    /// The linewidth of the optical transition is in SI units of Hz.
    pub fn calculate_for(
        dipole_beam_wavelength: f64,
        optical_transition_wavelength: f64,
        optical_transition_linewidth: f64,
    ) -> Polarizability {
        let transition_f = constant::C / optical_transition_wavelength;
        let dipole_f = constant::C / dipole_beam_wavelength;
        let prefactor = -3. * constant::PI * constant::C.powf(2.0)
            / (2. * (2. * constant::PI * transition_f).powf(3.0))
            * optical_transition_linewidth
            * -(1. / (transition_f - dipole_f) + 1. / (transition_f + dipole_f));
        Polarizability { prefactor }
    }
}

/// A system that attaches `DipoleLightIndex` components to entities which have `DipoleLight` but no index.
pub struct AttachIndexToDipoleLightSystem;
impl<'a> System<'a> for AttachIndexToDipoleLightSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, DipoleLight>,
        ReadStorage<'a, LaserIndex>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (ent, dipole_light, indices, updater): Self::SystemData) {
        for (ent, _, _) in (&ent, &dipole_light, !&indices).join() {
            updater.insert(ent, LaserIndex::default());
        }
    }
}

/// This plugin implements a dipole force that can be used to confine cold atoms.
/// 
/// See also [crate::dipole]
/// 
/// # Generic Arguments
/// 
/// * `N`: The maximum number of laser beams (must match the `LaserPlugin`).
pub struct DipolePlugin<const N : usize>;
impl<const N: usize> Plugin for DipolePlugin<N> {
    fn build(&self, builder: &mut crate::simulation::SimulationBuilder) {
        add_systems_to_dispatch::<N>(&mut builder.dispatcher_builder, &[]);
        register_components(&mut builder.world);
    }
    fn deps(&self) -> Vec::<Box<dyn Plugin>> {
        vec![Box::new(LaserPlugin::<{N}>)]
    }
}

/// Adds the systems required by the module to the dispatcher.
///
/// #Arguments
///
/// `builder`: the dispatch builder to modify
///
/// `deps`: any dependencies that must be completed before the systems run.
fn add_systems_to_dispatch<const N: usize>(
    builder: &mut DispatcherBuilder<'static, 'static>,
    deps: &[&str],
) {
    builder.add(
        force::ApplyDipoleForceSystem::<N>,
        "apply_dipole_force",
        &["sample_intensity_gradient"],
    );
    builder.add(
        crate::dipole::AttachIndexToDipoleLightSystem,
        "attach_dipole_index",
        deps,
    );
}

fn register_components(world: &mut World) {
    world.register::<DipoleLight>();
}
