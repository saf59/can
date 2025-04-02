// build.rs
use std::env;

fn main() {
    // Retrieve the target operating system environment variable set by Cargo.
    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .expect("CARGO_CFG_TARGET_OS environment variable not set by Cargo.");

    // Check if the target OS is "windows".
    if target_os == "windows" {
        // If it is Windows, instruct cargo to pass a cfg flag to rustc,
        // effectively enabling the "os_windows" feature for this build.
        // This uses the `rustc-cfg` directive.
        println!("cargo:rustc-cfg=feature=\"os_windows\"");
    }

    // Tell Cargo to rerun this build script only if build.rs itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    // Build script finishes. If the target was Windows, the feature="os_windows"
    // cfg flag will be active for the main crate compilation.
}
