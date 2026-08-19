use crate::{Error, Result};

/// Exact `floor(decimal * multiplier)` for a non-negative finite decimal
/// literal. No binary floating-point participates in material accounting.
pub(crate) fn floor_mul(text: &str, multiplier: u64) -> Result<u64> {
    let s = text.trim();
    if s.is_empty() {
        return Err(Error::Invalid("budget must be a decimal number".into()));
    }
    let s = s.strip_prefix('+').unwrap_or(s);
    if s.starts_with('-') {
        return Err(Error::Invalid(
            "budget must be finite and nonnegative".into(),
        ));
    }

    let (mantissa, exponent) = split_exponent(s)?;
    let mut digits = Vec::with_capacity(mantissa.len());
    let mut frac_digits: i128 = 0;
    let mut seen_dot = false;
    let mut seen_digit = false;
    for b in mantissa.bytes() {
        match b {
            b'0'..=b'9' => {
                seen_digit = true;
                digits.push(b - b'0');
                if seen_dot {
                    frac_digits += 1;
                }
            }
            b'.' if !seen_dot => seen_dot = true,
            _ => return Err(Error::Invalid("budget must be a decimal number".into())),
        }
    }
    if !seen_digit {
        return Err(Error::Invalid("budget must be a decimal number".into()));
    }

    let first_nonzero = digits.iter().position(|&d| d != 0);
    let Some(first_nonzero) = first_nonzero else {
        return Ok(0);
    };
    if multiplier == 0 {
        return Ok(0);
    }
    digits.drain(..first_nonzero);

    let product = mul_digits(&digits, multiplier);
    let scale = frac_digits - exponent;
    let integer_digits = if scale > 0 {
        let scale = usize::try_from(scale).unwrap_or(usize::MAX);
        if scale >= product.len() {
            return Ok(0);
        }
        &product[..product.len() - scale]
    } else {
        &product[..]
    };

    let appended_zeros = if scale < 0 {
        usize::try_from(-scale).unwrap_or(usize::MAX)
    } else {
        0
    };
    if integer_digits.len().saturating_add(appended_zeros) > 20 {
        return Err(Error::Invalid("budget is too large".into()));
    }

    let mut value = 0u64;
    for &digit in integer_digits {
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit as u64))
            .ok_or_else(|| Error::Invalid("budget is too large".into()))?;
    }
    for _ in 0..appended_zeros {
        value = value
            .checked_mul(10)
            .ok_or_else(|| Error::Invalid("budget is too large".into()))?;
    }
    Ok(value)
}

fn split_exponent(s: &str) -> Result<(&str, i128)> {
    let mut split = None;
    for (i, ch) in s.char_indices() {
        if ch == 'e' || ch == 'E' {
            if split.is_some() {
                return Err(Error::Invalid("budget must be a decimal number".into()));
            }
            split = Some(i);
        }
    }
    let Some(i) = split else {
        return Ok((s, 0));
    };
    let mantissa = &s[..i];
    let exp = &s[i + 1..];
    if exp.is_empty() {
        return Err(Error::Invalid("budget must be a decimal number".into()));
    }
    let exponent = parse_exponent(exp)?;
    Ok((mantissa, exponent))
}

fn parse_exponent(s: &str) -> Result<i128> {
    let (negative, digits) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        (false, rest)
    } else {
        (false, s)
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::Invalid("budget exponent is out of range".into()));
    }
    // Saturation is sufficient: once the exponent exceeds i128's practical
    // range the final result is necessarily either zero or too large for u64.
    let mut value = 0i128;
    for b in digits.bytes() {
        value = value.saturating_mul(10).saturating_add((b - b'0') as i128);
    }
    Ok(if negative {
        value.saturating_neg()
    } else {
        value
    })
}

fn mul_digits(digits: &[u8], multiplier: u64) -> Vec<u8> {
    let mut out = vec![0u8; digits.len() + 20];
    let mut write = out.len();
    let mut carry = 0u128;
    for &digit in digits.iter().rev() {
        let n = digit as u128 * multiplier as u128 + carry;
        write -= 1;
        out[write] = (n % 10) as u8;
        carry = n / 10;
    }
    while carry != 0 {
        write -= 1;
        out[write] = (carry % 10) as u8;
        carry /= 10;
    }
    out.drain(..write);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_floor() {
        assert_eq!(floor_mul("1.9", 24).unwrap(), 45);
        assert_eq!(floor_mul("1e-2", 100).unwrap(), 1);
        assert_eq!(floor_mul(".5", 25).unwrap(), 12);
        assert_eq!(floor_mul("100", 24).unwrap(), 2400);
        assert_eq!(floor_mul("1e-1000", 24).unwrap(), 0);
        assert!(floor_mul("1e1000", 24).is_err());
        assert_eq!(floor_mul("0e999999999999999999999", 24).unwrap(), 0);
    }
}
