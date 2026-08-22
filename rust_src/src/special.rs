//! Dependency-free special functions used by the Rust backend.

#![allow(clippy::excessive_precision)]

/*
 * This complementary error function is adapted from
 * FreeBSD /usr/src/lib/msun/src/s_erf.c.
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

const ERX: f64 = 8.45062911510467529297e-01;

// Coefficients for erf on [0, 0.84375].
const PP0: f64 = 1.28379167095512558561e-01;
const PP1: f64 = -3.25042107247001499370e-01;
const PP2: f64 = -2.84817495755985104766e-02;
const PP3: f64 = -5.77027029648944159157e-03;
const PP4: f64 = -2.37630166566501626084e-05;
const QQ1: f64 = 3.97917223959155352819e-01;
const QQ2: f64 = 6.50222499887672944485e-02;
const QQ3: f64 = 5.08130628187576562776e-03;
const QQ4: f64 = 1.32494738004321644526e-04;
const QQ5: f64 = -3.96022827877536812320e-06;

// Coefficients for erf around x = 1.
const PA0: f64 = -2.36211856075265944077e-03;
const PA1: f64 = 4.14856118683748331666e-01;
const PA2: f64 = -3.72207876035701323847e-01;
const PA3: f64 = 3.18346619901161753674e-01;
const PA4: f64 = -1.10894694282396677476e-01;
const PA5: f64 = 3.54783043256182359371e-02;
const PA6: f64 = -2.16637559486879084300e-03;
const QA1: f64 = 1.06420880400844228286e-01;
const QA2: f64 = 5.40397917702171048937e-01;
const QA3: f64 = 7.18286544141962662868e-02;
const QA4: f64 = 1.26171219808761642112e-01;
const QA5: f64 = 1.36370839120290507362e-02;
const QA6: f64 = 1.19844998467991074170e-02;

// Coefficients for erfc on [1.25, 1 / 0.35].
const RA0: f64 = -9.86494403484714822705e-03;
const RA1: f64 = -6.93858572707181764372e-01;
const RA2: f64 = -1.05586262253232909814e+01;
const RA3: f64 = -6.23753324503260060396e+01;
const RA4: f64 = -1.62396669462573470355e+02;
const RA5: f64 = -1.84605092906711035994e+02;
const RA6: f64 = -8.12874355063065934246e+01;
const RA7: f64 = -9.81432934416914548592e+00;
const SA1: f64 = 1.96512716674392571292e+01;
const SA2: f64 = 1.37657754143519042600e+02;
const SA3: f64 = 4.34565877475229228821e+02;
const SA4: f64 = 6.45387271733267880336e+02;
const SA5: f64 = 4.29008140027567833386e+02;
const SA6: f64 = 1.08635005541779435134e+02;
const SA7: f64 = 6.57024977031928170135e+00;
const SA8: f64 = -6.04244152148580987438e-02;

// Coefficients for erfc on [1 / 0.35, 28].
const RB0: f64 = -9.86494292470009928597e-03;
const RB1: f64 = -7.99283237680523006574e-01;
const RB2: f64 = -1.77579549177547519889e+01;
const RB3: f64 = -1.60636384855821916062e+02;
const RB4: f64 = -6.37566443368389627722e+02;
const RB5: f64 = -1.02509513161107724954e+03;
const RB6: f64 = -4.83519191608651397019e+02;
const SB1: f64 = 3.03380607434824582924e+01;
const SB2: f64 = 3.25792512996573918826e+02;
const SB3: f64 = 1.53672958608443695994e+03;
const SB4: f64 = 3.19985821950859553908e+03;
const SB5: f64 = 2.55305040643316442583e+03;
const SB6: f64 = 4.74528541206955367215e+02;
const SB7: f64 = -2.24409524465858183362e+01;

#[inline]
fn high_word(value: f64) -> u32 {
    (value.to_bits() >> 32) as u32
}

#[inline]
fn truncate_low_word(value: f64) -> f64 {
    f64::from_bits(value.to_bits() & 0xffff_ffff_0000_0000)
}

#[inline]
fn erfc_around_one(value: f64) -> f64 {
    let shifted = value.abs() - 1.0;
    let numerator = PA0
        + shifted
            * (PA1
                + shifted
                    * (PA2 + shifted * (PA3 + shifted * (PA4 + shifted * (PA5 + shifted * PA6)))));
    let denominator = 1.0
        + shifted
            * (QA1
                + shifted
                    * (QA2 + shifted * (QA3 + shifted * (QA4 + shifted * (QA5 + shifted * QA6)))));
    1.0 - ERX - numerator / denominator
}

#[inline]
fn polynomial(value: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |accumulator, coefficient| {
            accumulator * value + coefficient
        })
}

#[inline]
fn erfc_positive_tail(high_bits: u32, value: f64) -> f64 {
    if high_bits < 0x3ff4_0000 {
        return erfc_around_one(value);
    }

    let positive = value.abs();
    let reciprocal_square = 1.0 / (positive * positive);
    let (numerator, denominator) = if high_bits < 0x4006_db6d {
        (
            polynomial(reciprocal_square, &[RA0, RA1, RA2, RA3, RA4, RA5, RA6, RA7]),
            polynomial(
                reciprocal_square,
                &[1.0, SA1, SA2, SA3, SA4, SA5, SA6, SA7, SA8],
            ),
        )
    } else {
        (
            polynomial(reciprocal_square, &[RB0, RB1, RB2, RB3, RB4, RB5, RB6]),
            polynomial(reciprocal_square, &[1.0, SB1, SB2, SB3, SB4, SB5, SB6, SB7]),
        )
    };
    let truncated = truncate_low_word(positive);

    (-truncated * truncated - 0.5625).exp()
        * ((truncated - positive) * (truncated + positive) + numerator / denominator).exp()
        / positive
}

/// Return the complementary error function with close to full `f64` precision.
pub(crate) fn erfc(value: f64) -> f64 {
    let mut high_bits = high_word(value);
    let negative = high_bits >> 31 != 0;
    high_bits &= 0x7fff_ffff;

    if high_bits >= 0x7ff0_0000 {
        let limit = if negative { 2.0 } else { 0.0 };
        return limit + 1.0 / value;
    }

    if high_bits < 0x3feb_0000 {
        if high_bits < 0x3c70_0000 {
            return 1.0 - value;
        }
        let square = value * value;
        let numerator = PP0 + square * (PP1 + square * (PP2 + square * (PP3 + square * PP4)));
        let denominator =
            1.0 + square * (QQ1 + square * (QQ2 + square * (QQ3 + square * (QQ4 + square * QQ5))));
        let correction = numerator / denominator;
        if negative || high_bits < 0x3fd0_0000 {
            return 1.0 - (value + value * correction);
        }
        return 0.5 - (value - 0.5 + value * correction);
    }

    if high_bits < 0x403c_0000 {
        let tail = erfc_positive_tail(high_bits, value);
        return if negative { 2.0 - tail } else { tail };
    }

    if negative { 2.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_matches_reference_values() {
        let cases: [(f64, f64); 10] = [
            (-3.0, 1.999_977_909_503_001_5),
            (-1.0, 1.842_700_792_949_714_8),
            (-0.5, 1.520_499_877_813_046_5),
            (0.0, 1.0),
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_13),
            (2.0, 0.004_677_734_981_047_266),
            (3.0, 2.209_049_699_858_543_8e-5),
            (5.0, 1.537_459_794_428_035e-12),
            (8.0, 1.122_429_717_298_292_8e-29),
        ];

        for (value, expected) in cases {
            let actual = erfc(value);
            let tolerance = 16.0 * f64::EPSILON * expected.abs().max(f64::MIN_POSITIVE);
            assert!(
                (actual - expected).abs() <= tolerance,
                "erfc({value}) was {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn erfc_handles_special_values() {
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert!(erfc(f64::NAN).is_nan());
    }
}
