use std::{fs, io::Result, path::Path, process::Command, time::SystemTime};

fn main() -> Result<()> {
    compile_stb();
    compile_shaders()
}

fn compile_stb() {
    println!("cargo:rerun-if-changed=src/stb/");
    cc::Build::new()
        .file("src/stb/stb_image.c")
        .compile("stb_image");
    cc::Build::new()
        .file("src/stb/stb_truetype.c")
        .compile("stb_truetype");
}

fn compile_shaders() -> Result<()> {
    println!("cargo:rerun-if-changed=shaders");

    let shaders_dir = Path::new("shaders");
    let compiled_shaders_dir = Path::new("target/shaders");

    // Collect all shader files.
    let mut shaders = Vec::new();
    visit_dirs(shaders_dir, &mut |file| {
        if let Some(ext) = file.extension() {
            match ext.to_str().unwrap() {
                "vert" | "frag" => shaders.push(file.to_path_buf()),
                _ => {}
            }
        }
    })?;

    // Compiler shader if it's been updated.
    for src in shaders {
        let relative_path = src.strip_prefix(shaders_dir).unwrap();
        let mut spv_path = compiled_shaders_dir.join(relative_path);
        spv_path.set_extension(format!(
            "{}.spv",
            src.extension().unwrap().to_str().unwrap()
        ));

        if let Some(parent) = spv_path.parent() {
            fs::create_dir_all(parent)?;
        }

        if need_recompile(&src, &spv_path) {
            println!("cargo:info=glslc compiling {:?}", src);
            let status = Command::new("glslc")
                .arg(&src)
                .arg("-o")
                .arg(&spv_path)
                .status()
                .expect("failed to spawn glslc");

            if !status.success() {
                panic!("glslc failed for {:?} -> {:?}", src, spv_path);
            }
        }
    }

    Ok(())
}

fn visit_dirs(dir: &Path, f: &mut impl FnMut(&Path)) -> Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, f)?;
            } else {
                f(&path);
            }
        }
    }
    Ok(())
}

fn need_recompile(src: &Path, out: &Path) -> bool {
    fn modified_time(p: &Path) -> Result<SystemTime> {
        Ok(fs::metadata(p)?.modified()?)
    }
    let src_time = modified_time(src).ok();
    let out_time = modified_time(out).ok();
    src_time > out_time
}
