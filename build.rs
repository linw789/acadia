use regex::Regex;
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
                "slang" => shaders.push(file.to_path_buf()),
                _ => {}
            }
        }
    })?;

    // Compile shader if it's been updated.
    for src_file in shaders {
        let relative_path = src_file.strip_prefix(shaders_dir).unwrap();
        let mut vert_spv_out_path = compiled_shaders_dir.join(relative_path);
        let mut frag_spv_out_path = vert_spv_out_path.clone();
        vert_spv_out_path.set_extension("vert.spv");
        frag_spv_out_path.set_extension("frag.spv");

        if let Some(parent) = vert_spv_out_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let src_file_str = fs::read_to_string(&src_file).unwrap();

        for (spv_path, stage) in [
            (vert_spv_out_path, "vertex"),
            (frag_spv_out_path, "fragment"),
        ] {
            let has_shader = {
                let slang_attribute_regex =
                    Regex::new(&format!(r#"\[shader\(\s*\"{}\"\s*\)\]"#, stage)).unwrap();
                slang_attribute_regex.is_match(&src_file_str)
            };
            if has_shader && need_recompile(&src_file, &spv_path) {
                println!("cargo:info=slangc compiling {:?}", src_file);
                let status = Command::new("slangc")
                    .arg(&src_file)
                    .arg("-profile")
                    .arg("glsl_450")
                    .arg("-target")
                    .arg("spirv")
                    .arg("-entry")
                    .arg(format!("{}_main", stage))
                    .arg("-o")
                    .arg(&spv_path)
                    .status()
                    .expect("failed to spawn slangc");

                if !status.success() {
                    panic!("slangc failed for {:?} -> {:?}", src_file, spv_path);
                }
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
