//! The operating system as the search sees it: environment variables, files and a clock.
//!
//! Natively these are exactly `std::env`, `std::fs` and `std::time::Instant` (re-exports, so
//! native behaviour is unchanged). In the browser (wasm32-unknown-unknown) std has none of them,
//! so: the environment is a map of this WebAssembly instance (one instance is one "process":
//! the SM64Coin web miner runs each campaign step in a fresh instance, as the native miner runs
//! each in a fresh `rrtrun`), files live on the JavaScript side (`globalThis.sm64fs`, shared by
//! every instance of a worker, like one machine's file system), and the clock is `web-time`'s.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

/// This process's id (`std::process::id`); 0 in the browser, which has no processes ids (std
/// panics there).
pub fn process_id() -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    return std::process::id();
    #[cfg(target_arch = "wasm32")]
    0
}
#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub mod env {
    pub use std::env::{VarError, args, current_dir, remove_var, set_var, temp_dir, var, var_os, vars};
}

#[cfg(not(target_arch = "wasm32"))]
pub mod fs {
    pub use std::fs::*;
}

#[cfg(target_arch = "wasm32")]
pub mod env {
    use std::collections::BTreeMap;
    use std::ffi::{OsStr, OsString};
    use std::sync::Mutex;
    pub use std::env::{VarError, args};

    static VARS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

    pub fn var<K: AsRef<OsStr>>(key: K) -> Result<String, VarError> {
        VARS.lock().unwrap().get(&*key.as_ref().to_string_lossy()).cloned().ok_or(VarError::NotPresent)
    }
    pub fn var_os<K: AsRef<OsStr>>(key: K) -> Option<OsString> {
        var(key).ok().map(OsString::from)
    }
    /// # Safety
    /// None needed here (a map of this instance); unsafe to match `std::env::set_var`.
    pub unsafe fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
        VARS.lock().unwrap().insert(key.as_ref().to_string_lossy().into_owned(), value.as_ref().to_string_lossy().into_owned());
    }
    /// # Safety
    /// As [`set_var`].
    pub unsafe fn remove_var<K: AsRef<OsStr>>(key: K) {
        VARS.lock().unwrap().remove(&*key.as_ref().to_string_lossy());
    }
    pub fn vars() -> impl Iterator<Item = (String, String)> {
        VARS.lock().unwrap().clone().into_iter()
    }
    pub fn temp_dir() -> std::path::PathBuf {
        "/tmp".into()
    }
}

#[cfg(target_arch = "wasm32")]
pub mod fs {
    //! Files on the JavaScript side: `globalThis.sm64fs` = {read(path) -> Uint8Array|undefined,
    //! write(path, bytes), remove(path) -> bool, list(dir) -> "name\nname..."} (see the web
    //! miner's worker). Directories exist implicitly.
    use std::io::{self, Read, Write};
    use std::path::{Path, PathBuf};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = sm64fs, js_name = read)]
        fn js_read(path: &str) -> Option<Vec<u8>>;
        #[wasm_bindgen(js_namespace = sm64fs, js_name = write)]
        fn js_write(path: &str, data: &[u8]);
        #[wasm_bindgen(js_namespace = sm64fs, js_name = remove)]
        fn js_remove(path: &str) -> bool;
        #[wasm_bindgen(js_namespace = sm64fs, js_name = list)]
        fn js_list(dir: &str) -> String;
    }

    fn key(p: &Path) -> String {
        p.to_string_lossy().into_owned()
    }
    fn not_found(p: &Path) -> io::Error {
        io::Error::new(io::ErrorKind::NotFound, format!("{}: not found", p.display()))
    }

    pub fn read<P: AsRef<Path>>(p: P) -> io::Result<Vec<u8>> {
        js_read(&key(p.as_ref())).ok_or_else(|| not_found(p.as_ref()))
    }
    pub fn read_to_string<P: AsRef<Path>>(p: P) -> io::Result<String> {
        String::from_utf8(read(p)?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
    pub fn write<P: AsRef<Path>, C: AsRef<[u8]>>(p: P, data: C) -> io::Result<()> {
        js_write(&key(p.as_ref()), data.as_ref());
        Ok(())
    }
    pub fn create_dir_all<P: AsRef<Path>>(_: P) -> io::Result<()> {
        Ok(())
    }
    pub fn create_dir<P: AsRef<Path>>(_: P) -> io::Result<()> {
        Ok(())
    }
    pub fn remove_file<P: AsRef<Path>>(p: P) -> io::Result<()> {
        if js_remove(&key(p.as_ref())) { Ok(()) } else { Err(not_found(p.as_ref())) }
    }
    pub fn remove_dir_all<P: AsRef<Path>>(p: P) -> io::Result<()> {
        for e in read_dir(&p)? {
            let _ = remove_file(e?.path());
        }
        Ok(())
    }
    pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
        let data = read(&from)?;
        write(&to, data)?;
        let _ = remove_file(from);
        Ok(())
    }
    pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
        let data = read(from)?;
        let n = data.len() as u64;
        write(to, data)?;
        Ok(n)
    }
    pub fn canonicalize<P: AsRef<Path>>(p: P) -> io::Result<PathBuf> {
        Ok(p.as_ref().to_path_buf())
    }
    pub fn metadata<P: AsRef<Path>>(p: P) -> io::Result<Metadata> {
        Ok(Metadata { len: read(p)?.len() as u64 })
    }
    pub struct Metadata {
        len: u64,
    }
    impl Metadata {
        pub fn len(&self) -> u64 {
            self.len
        }
        pub fn is_file(&self) -> bool {
            true
        }
        pub fn is_dir(&self) -> bool {
            false
        }
    }

    pub struct DirEntry {
        path: PathBuf,
    }
    impl DirEntry {
        pub fn path(&self) -> PathBuf {
            self.path.clone()
        }
        pub fn file_name(&self) -> std::ffi::OsString {
            self.path.file_name().unwrap_or_default().to_os_string()
        }
        pub fn metadata(&self) -> io::Result<Metadata> {
            metadata(&self.path)
        }
    }
    pub fn read_dir<P: AsRef<Path>>(p: P) -> io::Result<std::vec::IntoIter<io::Result<DirEntry>>> {
        let dir = p.as_ref().to_path_buf();
        let names = js_list(&key(&dir));
        Ok(names.lines().filter(|n| !n.is_empty()).map(|n| Ok(DirEntry { path: dir.join(n) })).collect::<Vec<_>>().into_iter())
    }

    /// A file: reads see its contents when opened, writes go to the JavaScript side on flush
    /// and drop.
    pub struct File {
        path: String,
        data: Vec<u8>,
        pos: usize,
        dirty: bool,
    }
    impl File {
        pub fn create<P: AsRef<Path>>(p: P) -> io::Result<File> {
            Ok(File { path: key(p.as_ref()), data: Vec::new(), pos: 0, dirty: true })
        }
        pub fn open<P: AsRef<Path>>(p: P) -> io::Result<File> {
            Ok(File { path: key(p.as_ref()), data: read(p)?, pos: 0, dirty: false })
        }
        pub fn options() -> OpenOptions {
            OpenOptions::new()
        }
    }
    impl Write for File {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.data.extend_from_slice(buf);
            self.dirty = true;
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.dirty {
                js_write(&self.path, &self.data);
                self.dirty = false;
            }
            Ok(())
        }
    }
    impl Read for File {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }
    impl Drop for File {
        fn drop(&mut self) {
            let _ = self.flush();
        }
    }

    #[derive(Default, Clone)]
    pub struct OpenOptions {
        append: bool,
        truncate: bool,
    }
    impl OpenOptions {
        pub fn new() -> OpenOptions {
            OpenOptions::default()
        }
        pub fn read(&mut self, _: bool) -> &mut Self {
            self
        }
        pub fn write(&mut self, _: bool) -> &mut Self {
            self
        }
        pub fn create(&mut self, _: bool) -> &mut Self {
            self
        }
        pub fn append(&mut self, a: bool) -> &mut Self {
            self.append = a;
            self
        }
        pub fn truncate(&mut self, t: bool) -> &mut Self {
            self.truncate = t;
            self
        }
        pub fn open<P: AsRef<Path>>(&self, p: P) -> io::Result<File> {
            let data = if self.append && !self.truncate { js_read(&key(p.as_ref())).unwrap_or_default() } else { Vec::new() };
            Ok(File { path: key(p.as_ref()), pos: data.len(), data, dirty: false })
        }
    }
}
