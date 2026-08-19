use std::collections::BTreeMap;

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

pub(crate) fn parse(input: &str) -> Result<Value> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let value = p.value()?;
    p.ws();
    if p.pos != p.bytes.len() {
        return Err(Error::Json(format!(
            "unexpected trailing JSON at byte {}",
            p.pos
        )));
    }
    Ok(value)
}

pub(crate) fn quote(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c < '\u{20}' => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Result<Value> {
        self.ws();
        match self.bytes.get(self.pos).copied() {
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(other) => Err(Error::Json(format!(
                "unexpected byte {other:?} at {}",
                self.pos
            ))),
            None => Err(Error::Json("unexpected end of JSON".into())),
        }
    }

    fn literal(&mut self, expected: &[u8]) -> Result<()> {
        let end = self.pos + expected.len();
        if self.bytes.get(self.pos..end) != Some(expected) {
            return Err(Error::Json(format!("invalid literal at byte {}", self.pos)));
        }
        self.pos = end;
        Ok(())
    }

    fn array(&mut self) -> Result<Value> {
        self.pos += 1;
        let mut out = Vec::new();
        self.ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Value::Array(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(Error::Json(format!(
                        "expected ',' or ']' at byte {}",
                        self.pos
                    )))
                }
            }
        }
        Ok(Value::Array(out))
    }

    fn object(&mut self) -> Result<Value> {
        self.pos += 1;
        let mut out = BTreeMap::new();
        self.ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Value::Object(out));
        }
        loop {
            self.ws();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return Err(Error::Json(format!(
                    "expected object key at byte {}",
                    self.pos
                )));
            }
            let key = self.string()?;
            self.ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return Err(Error::Json(format!("expected ':' at byte {}", self.pos)));
            }
            self.pos += 1;
            let value = self.value()?;
            if out.insert(key, value).is_some() {
                return Err(Error::Json("duplicate object key".into()));
            }
            self.ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(Error::Json(format!(
                        "expected ',' or '}}' at byte {}",
                        self.pos
                    )))
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn string(&mut self) -> Result<String> {
        debug_assert_eq!(self.bytes.get(self.pos), Some(&b'"'));
        self.pos += 1;
        let mut out = String::new();
        let mut segment = self.pos;
        loop {
            let Some(&b) = self.bytes.get(self.pos) else {
                return Err(Error::Json("unterminated JSON string".into()));
            };
            match b {
                b'"' => {
                    out.push_str(
                        std::str::from_utf8(&self.bytes[segment..self.pos])
                            .map_err(|_| Error::Json("invalid UTF-8 in JSON string".into()))?,
                    );
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    out.push_str(
                        std::str::from_utf8(&self.bytes[segment..self.pos])
                            .map_err(|_| Error::Json("invalid UTF-8 in JSON string".into()))?,
                    );
                    self.pos += 1;
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| Error::Json("unterminated escape".into()))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            if (0xD800..=0xDBFF).contains(&first) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(Error::Json(
                                        "high surrogate without low surrogate".into(),
                                    ));
                                }
                                self.pos += 2;
                                let second = self.hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&second) {
                                    return Err(Error::Json("invalid low surrogate".into()));
                                }
                                let cp = 0x10000
                                    + (((first - 0xD800) as u32) << 10)
                                    + (second - 0xDC00) as u32;
                                out.push(
                                    char::from_u32(cp).ok_or_else(|| {
                                        Error::Json("invalid unicode escape".into())
                                    })?,
                                );
                            } else if (0xDC00..=0xDFFF).contains(&first) {
                                return Err(Error::Json("unpaired low surrogate".into()));
                            } else {
                                out.push(
                                    char::from_u32(first as u32).ok_or_else(|| {
                                        Error::Json("invalid unicode escape".into())
                                    })?,
                                );
                            }
                        }
                        _ => return Err(Error::Json("invalid string escape".into())),
                    }
                    segment = self.pos;
                }
                0x00..=0x1f => return Err(Error::Json("control byte in JSON string".into())),
                _ => self.pos += 1,
            }
        }
    }

    fn hex4(&mut self) -> Result<u16> {
        let end = self.pos + 4;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::Json("short unicode escape".into()))?;
        let mut n = 0u16;
        for &b in bytes {
            n = n * 16
                + match b {
                    b'0'..=b'9' => (b - b'0') as u16,
                    b'a'..=b'f' => (b - b'a' + 10) as u16,
                    b'A'..=b'F' => (b - b'A' + 10) as u16,
                    _ => return Err(Error::Json("invalid unicode escape".into())),
                };
        }
        self.pos = end;
        Ok(n)
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(Error::Json(format!("invalid number at byte {start}"))),
        }
        if self.bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            let frac = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == frac {
                return Err(Error::Json(format!("invalid number at byte {start}")));
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == exp {
                return Err(Error::Json(format!("invalid number at byte {start}")));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        let n: f64 = text
            .parse()
            .map_err(|_| Error::Json(format!("invalid number at byte {start}")))?;
        if !n.is_finite() {
            return Err(Error::Json("non-finite JSON number".into()));
        }
        Ok(Value::Number(text.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundish() {
        let v = parse(r#"{"x":[1,-2.5e2,true,null,"é\\n"]}"#).unwrap();
        let Value::Object(o) = v else { panic!() };
        assert!(o.contains_key("x"));
    }
}
