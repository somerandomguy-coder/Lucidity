//! Hand-written JSON writer (no serde).

pub struct Json {
    s: String,
    first: bool,
}

fn esc(out: &mut String, v: &str) {
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn fmt_num(v: f64) -> String {
    if !v.is_finite() {
        return "null".into();
    }
    // 5 significant decimals is plenty for display and the scene
    let r = (v * 1e5).round() / 1e5;
    let s = format!("{}", r);
    if s == "-0" { "0".into() } else { s }
}

impl Json {
    pub fn obj() -> Json {
        Json { s: String::from("{"), first: true }
    }
    fn key(&mut self, k: &str) {
        if !self.first {
            self.s.push(',');
        }
        self.first = false;
        esc(&mut self.s, k);
        self.s.push(':');
    }
    pub fn num(&mut self, k: &str, v: f64) -> &mut Self {
        self.key(k);
        self.s.push_str(&fmt_num(v));
        self
    }
    pub fn opt_num(&mut self, k: &str, v: Option<f64>) -> &mut Self {
        self.key(k);
        match v {
            Some(v) => self.s.push_str(&fmt_num(v)),
            None => self.s.push_str("null"),
        }
        self
    }
    pub fn str(&mut self, k: &str, v: &str) -> &mut Self {
        self.key(k);
        esc(&mut self.s, v);
        self
    }
    pub fn opt_str(&mut self, k: &str, v: Option<&str>) -> &mut Self {
        self.key(k);
        match v {
            Some(v) => esc(&mut self.s, v),
            None => self.s.push_str("null"),
        }
        self
    }
    pub fn bool(&mut self, k: &str, v: bool) -> &mut Self {
        self.key(k);
        self.s.push_str(if v { "true" } else { "false" });
        self
    }
    pub fn nums(&mut self, k: &str, v: &[f64]) -> &mut Self {
        self.key(k);
        self.s.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                self.s.push(',');
            }
            self.s.push_str(&fmt_num(*x));
        }
        self.s.push(']');
        self
    }
    pub fn strs(&mut self, k: &str, v: &[&str]) -> &mut Self {
        self.key(k);
        self.s.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                self.s.push(',');
            }
            esc(&mut self.s, x);
        }
        self.s.push(']');
        self
    }
    /// Insert an already-serialized JSON value.
    pub fn raw(&mut self, k: &str, v: &str) -> &mut Self {
        self.key(k);
        self.s.push_str(v);
        self
    }
    pub fn end(&mut self) -> String {
        let mut s = std::mem::take(&mut self.s);
        s.push('}');
        s
    }
}
