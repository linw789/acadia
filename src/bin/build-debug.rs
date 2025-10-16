use std::{fs, io::Result, path::Path, time::SystemTime};

fn main() -> Result<()> {
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

    // Compiler shader if it's been updated.
    for src_file in shaders {
        let relative_path = src_file.strip_prefix(shaders_dir).unwrap();
        let mut vert_spv_path = compiled_shaders_dir.join(relative_path);
        let mut frag_spv_path = vert_spv_path.clone();
        vert_spv_path.set_extension("vert.spv");
        frag_spv_path.set_extension("frag.spv");

        if let Some(parent) = vert_spv_path.parent() {
            fs::create_dir_all(parent)?;
        }

        for (spv_path, stage) in [(vert_spv_path, "vertex"), (frag_spv_path, "fragment")] {
            if need_recompile(&src_file, &spv_path) {
                println!(
                    "[DEBUG] slangc compiling: {:?}, stage: {}, output: {:?}",
                    src_file, stage, spv_path
                );
            } else {
                println!("[DEBUG] no need to compile. stage: {}", stage);
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
    println!("[DEBUG] src_time: {:?}, out_time: {:?}", src_time, out_time);
    src_time > out_time
}
