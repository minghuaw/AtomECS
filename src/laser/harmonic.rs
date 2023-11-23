use nalgebra::Vector3;

use crate::{atom::Position, constant::PI, maths};

use super::{frame::Frame, gaussian::{GaussianBeam, CircularMask}};

#[allow(non_snake_case)]
pub(crate) fn gaussian_beam_intensity_first_order_taylor_expansion(
    beam: &GaussianBeam,
    pos: &Position,
    mask: Option<&CircularMask>,
    frame: Option<&Frame>,
) -> f64 {
    let x0 = beam.intersection.x;
    let y0 = beam.intersection.y;

    // Relative distance between the point and the beam intersection
    let (dz, relative_distance) = match frame {
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

    let we = beam.e_radius;
    let zr = beam.rayleigh_range;

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

    let z2_over_zr2_plus_one = dz.powi(2) / zr.powi(2) + 1.0;

    let sigma = f64::exp(
        -(2.0 * x0.powi(2) + 2.0 * y0.powi(2)) / (2.0 * we.powi(2) * z2_over_zr2_plus_one),
    );

    let zeroth_order = (P * sigma) / (we.powi(2) * PI * z2_over_zr2_plus_one);
    let x_first_order = -(2.0 * P * x0 * sigma * (relative_distance))
        / (we.powi(4) * PI * z2_over_zr2_plus_one.powi(2));
    let y_first_order = -(2.0 * P * y0 * sigma * (relative_distance))
        / (we.powi(4) * PI * z2_over_zr2_plus_one.powi(2));

    zeroth_order + x_first_order + y_first_order
}

#[allow(non_snake_case)]
pub(crate) fn gaussian_beam_intensity_gradient_first_order_taylor_expansion(
    beam: &GaussianBeam,
    pos: &Position,
    reference_frame: &Frame,
) -> Vector3<f64> {
    let rela_coord = pos.pos - beam.intersection;

    // ellipticity treatment
    let semi_major_axis = 1.0 / (1.0 - beam.ellipticity.powf(2.0)).powf(0.5);

    let dx = rela_coord.dot(&reference_frame.x_vector) / semi_major_axis.powf(0.5);
    let dy = rela_coord.dot(&reference_frame.y_vector) * semi_major_axis.powf(0.5);
    let dz = rela_coord.dot(&beam.direction);

    let x0 = beam.intersection.x;
    let y0 = beam.intersection.y;

    let zr = beam.rayleigh_range;
    let we = beam.e_radius;
    let P = beam.power;

    let sigma_6 = dz.powi(2) / zr.powi(2) + 1.0;
    let sigma_5 = 2.0 * x0.powi(2) + 2.0 * y0.powi(2);
    let sigma_4 = we.powi(6) * PI * sigma_6.powi(3);
    let sigma_3 = we.powi(4) * PI * sigma_6.powi(2);
    let sigma_2 = we.powi(4) * zr.powi(2) * PI * sigma_6.powi(3);
    let sigma_1 = f64::exp(-sigma_5 / (2.0 * we.powi(2) * sigma_6));

    let gx = (4.0 * P * x0 * y0 * sigma_1 * dy) / sigma_4
        - (2.0 * P * x0 * sigma_1) / sigma_3
        - (2.0 * P * dx * (sigma_1 - (2.0 * x0.powi(2) * sigma_1) / (we.powi(2) * sigma_6)))
            / sigma_3;
    let gy = (4.0 * P * x0 * y0 * sigma_1 * dx) / sigma_4
        - (2.0 * P * y0 * sigma_1) / sigma_3
        - (2.0 * P * dy * (sigma_1 - (2.0 * y0.powi(2) * sigma_1) / (we.powi(2) * sigma_6)))
            / sigma_3;
    let gz_1 = dx
        * ((P
            * dz
            * (4.0 * x0 * sigma_1 - (2.0 * x0 * sigma_1 * sigma_5) / (we.powi(2) * sigma_6))
            / sigma_2)
            + (4.0 * P * x0 * dz * sigma_1) / sigma_2);
    let gz_2 = dy
        * ((P
            * dz
            * (4.0 * y0 * sigma_1 - (2.0 * y0 * sigma_1 * sigma_5) / (we.powi(2) * sigma_6))
            / sigma_2)
            + (4.0 * P * y0 * dz * sigma_1) / sigma_2);
    let gz_3 = - 2.0 * P * dz * sigma_1 / (we.powi(2) * zr.powi(2) * PI * sigma_6.powi(2));
    let gz_4 = P * dz * sigma_1 * sigma_5 / sigma_2;
    let gz = gz_1 + gz_2 + gz_3 + gz_4;

    reference_frame.x_vector * gx + reference_frame.y_vector * gy + beam.direction * gz
}
