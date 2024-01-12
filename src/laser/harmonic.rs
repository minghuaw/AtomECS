use nalgebra::Vector3;

use crate::{atom::Position, constant::PI, maths};

use super::{
    frame::Frame,
    gaussian::{CircularMask, GaussianBeam, CircularAperture},
};

#[allow(non_snake_case)]
pub(crate) fn gaussian_beam_intensity_first_order_taylor_expansion(
    beam: &GaussianBeam,
    pos: &Position,
    mask: Option<&CircularMask>,
    aperture: Option<&CircularAperture>,
    frame: Option<&Frame>,
) -> f64 {
    // Relative distance between the point and the beam intersection
    let (z, relative_distance) = match frame {
        // checking if frame is given (for calculating ellipticity)
        Some(frame) => {
            let (x, y, z) = maths::get_relative_coordinates_line_point(
                &pos.pos,
                &beam.intersection,
                &beam.direction,
                frame,
            );
            let semi_major_axis = 1.0 / (1.0 - beam.ellipticity.powf(2.0)).powf(0.5);

            // the factor (1.0 / semi_major_axis) is necessary so the overall power of the beam is not changed.
            (
                z,
                // (1.0 / semi_major_axis) * ((x).powf(2.0) + (y * semi_major_axis).powf(2.0)),
                f64::sqrt(
                    (1.0 / semi_major_axis) * ((x).powf(2.0) + (y * semi_major_axis).powf(2.0)),
                ),
            )
        }
        // ellipticity will be ignored (i.e. treated as zero) if no `Frame` is supplied.
        None => {
            let (distance, z) = maths::get_minimum_distance_line_point(
                &pos.pos,
                &beam.intersection,
                &beam.direction,
            );
            (z, distance)
        }
    };
    let P = match mask {
        Some(mask) => {
            if relative_distance < mask.radius {
                0.0
            } else {
                beam.power
            }
        }
        None => beam.power,
    };
    let P = match aperture {
        Some(aperture) => {
            if relative_distance > aperture.radius {
                0.0
            } else {
                P
            }
        }
        None => P,
    };
    let w_e = beam.e_radius;
    let z_r = beam.rayleigh_range;
    let x = relative_distance;
    let y = relative_distance;

    (P * 1.0 / (w_e * w_e)) / (PI * ((z * z) * 1.0 / (z_r * z_r) + 1.0))
        - (P * 1.0 / (w_e * w_e * w_e * w_e) * (x * x) * 1.0
            / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 2.0))
            / PI
        - (P * 1.0 / (w_e * w_e * w_e * w_e) * (y * y) * 1.0
            / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 2.0))
            / PI
}

#[allow(non_snake_case)]
pub(crate) fn gaussian_beam_intensity_gradient_first_order_taylor_expansion(
    beam: &GaussianBeam,
    pos: &Position,
    mask: Option<&CircularMask>,
    aperture: Option<&CircularAperture>,
    reference_frame: &Frame,
) -> Vector3<f64> {
    let rela_coord = pos.pos - beam.intersection;

    // ellipticity treatment
    let semi_major_axis = 1.0 / (1.0 - beam.ellipticity.powf(2.0)).powf(0.5);

    let x = rela_coord.dot(&reference_frame.x_vector) / semi_major_axis.powf(0.5);
    let y = rela_coord.dot(&reference_frame.y_vector) * semi_major_axis.powf(0.5);
    let z = rela_coord.dot(&beam.direction);

    let P = beam.power;
    let w_e = beam.e_radius;
    let z_r = beam.rayleigh_range;

    let gx = (P * 1.0 / (w_e * w_e * w_e * w_e) * x * 1.0
        / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 2.0)
        * -2.0)
        / PI;
    let gy = (P * 1.0 / (w_e * w_e * w_e * w_e) * y * 1.0
        / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 2.0)
        * -2.0)
        / PI;
    let gz = (P * 1.0 / (w_e * w_e) * z * 1.0 / (z_r * z_r) * 1.0
        / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 2.0)
        * -2.0)
        / PI
        + (P * 1.0 / (w_e * w_e * w_e * w_e) * (x * x) * z * 1.0 / (z_r * z_r) * 1.0
            / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 3.0)
            * 4.0)
            / PI
        + (P * 1.0 / (w_e * w_e * w_e * w_e) * (y * y) * z * 1.0 / (z_r * z_r) * 1.0 * 4.0)
            / f64::powf((z * z) * 1.0 / (z_r * z_r) + 1.0, 3.0)
            / PI;

    let g = reference_frame.x_vector * gx + reference_frame.y_vector * gy + beam.direction * gz;

    let g = match mask {
        Some(mask) => {
            if x * x + y * y < mask.radius * mask.radius {
                Vector3::new(0.0, 0.0, 0.0)
            } else {
                g
            }
        }
        None => g,
    };

    let g = match aperture {
        Some(aperture) => {
            if x * x + y * y > aperture.radius * aperture.radius {
                Vector3::new(0.0, 0.0, 0.0)
            } else {
                g
            }
        }
        None => g,
    };

    g
}
