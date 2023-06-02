#[cfg(feature = "ffmpeg")]
mod ffmpeg_dep_libs {
    use std::path::PathBuf;

    const WASI_SDK: &'static str = "WASI_SDK";
    const WASI_SYSROOT: &'static str = "WASI_SYSROOT";
    const CLANG_RT: &'static str = "CLANG_RT";

    const DEFAULT_WASI_SDK: &'static str = "/opt/wasi-sdk";
    const CLANG_RT_LIB_NAME: &'static str = "clang_rt.builtins-wasm32";
    const WASI_CLOCK_LIB_NAME: &'static str = "wasi-emulated-process-clocks";

    fn check_lib_exists(mut dir: PathBuf, lib: &str) -> bool {
        let filename = if cfg!(windows) {
            String::from(lib) + ".lib"
        } else if cfg!(unix) {
            String::from("lib") + lib + ".a"
        } else {
            panic!("Unsupported host os");
        };

        dir.push(filename);
        eprintln!("cargo:warning={}", dir.display());
        dir.exists() && !dir.is_dir()
    }

    fn check_wasi_lib_path(search_dir: PathBuf, lib_name: &str) -> Option<PathBuf> {
        if !search_dir.is_dir() {
            return None;
        }
        // ${dir}/libxxx.a
        if check_lib_exists(search_dir.clone(), lib_name) {
            return Some(search_dir);
        }

        const VERSION_LIST: &[&str] = &["", "15.0.7", "16"];
        for version in VERSION_LIST {
            let mut dir = search_dir.clone();
            dir.push(version);

            dir.push("lib");
            // ${dir}/${version}/lib/libxxx.a
            if check_lib_exists(dir.clone(), lib_name) {
                return Some(dir);
            }

            dir.push("wasm32-wasi");
            // ${dir}/${version}/lib/wasm32-wasi/libxxx.a
            if check_lib_exists(dir.clone(), lib_name) {
                return Some(dir);
            }
            dir.pop();

            dir.push("wasi");
            // ${dir}/${version}/lib/wasi/libxxx.a
            if check_lib_exists(dir.clone(), lib_name) {
                return Some(dir);
            }
        }
        return None;
    }

    fn find_wasi_library_in_wasi_sdk(
        mut wasi_sdk: PathBuf,
        subdir: &PathBuf,
        libname: &str,
    ) -> Option<PathBuf> {
        wasi_sdk.push(&subdir);
        check_wasi_lib_path(wasi_sdk, libname)
    }

    fn find_wasi_library(env_key: &str, lib_name: &str, subdir_in_wasi_sdk: PathBuf) -> PathBuf {
        if let Ok(p) = std::env::var(env_key) {
            if let Some(p) = check_wasi_lib_path(PathBuf::from(p.as_str()), lib_name) {
                return p;
            } else {
                println!(
                    "cargo:warning=environment variable `{}`=`{}` is not a valid dir.",
                    env_key, p
                );
            }
        }
        if let Ok(p) = std::env::var(WASI_SDK) {
            if let Some(p) = find_wasi_library_in_wasi_sdk(
                PathBuf::from(p.as_str()),
                &subdir_in_wasi_sdk,
                lib_name,
            ) {
                return p;
            } else {
                println!(
                    "cargo:warning=environment variable `{}`=`{}` is not a valid dir.",
                    WASI_SDK, p
                );
            }
        }
        #[cfg(unix)]
        if let Some(p) = find_wasi_library_in_wasi_sdk(
            PathBuf::from(DEFAULT_WASI_SDK),
            &subdir_in_wasi_sdk,
            lib_name,
        ) {
            return p;
        }

        let mut search_list = Vec::new();
        let p = PathBuf::from(format!("${}", env_key));

        let mut t = p.clone();
        t.push(lib_name);
        search_list.push(t.display().to_string());

        let mut t = p.clone();
        t.push("lib");
        t.push(lib_name);
        search_list.push(t.display().to_string());

        let mut t = p.clone();
        t.push("lib");
        t.push("wasi");
        t.push(lib_name);
        search_list.push(t.display().to_string());

        let mut t = p.clone();
        t.push("lib");
        t.push("wasm32-wasi");
        t.push(lib_name);
        search_list.push(t.display().to_string());
        panic!(
            "Cannot find wasi-sysroot, please set `{}` or `{}` to change it.\nSearch list = {:?}",
            env_key, WASI_SDK, search_list
        );
    }

    fn find_wasi_clock_lib() -> PathBuf {
        let mut subdir = PathBuf::from("share");
        subdir.push("wasi-sysroot");
        find_wasi_library(WASI_SYSROOT, WASI_CLOCK_LIB_NAME, subdir)
    }

    fn find_clang_rt_path() -> PathBuf {
        let mut subdir = PathBuf::from("lib");
        subdir.push("clang");
        find_wasi_library(CLANG_RT, CLANG_RT_LIB_NAME, subdir)
    }

    pub fn main() {
        println!("cargo:rerun-if-env-changed={}", WASI_SDK);
        println!("cargo:rerun-if-env-changed={}", WASI_SYSROOT);
        println!("cargo:rerun-if-env-changed={}", CLANG_RT);

        println!(
            "cargo:rustc-link-search=native={}",
            find_clang_rt_path().display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            find_wasi_clock_lib().display()
        );
        println!(
            "cargo:rustc-flags=-l{} -l{}",
            CLANG_RT_LIB_NAME, WASI_CLOCK_LIB_NAME
        );
    }
}

fn main() {
    #[cfg(feature = "ffmpeg")]
    ffmpeg_dep_libs::main();
}
