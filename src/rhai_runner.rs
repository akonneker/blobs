use rhai::{Engine, EvalAltResult};
use rhai::packages::Package;    // needed for 'Package' trait
use rhai_rand::RandomPackage;
use std::env;
use std::fs;
use std::process;

fn main() -> Result<(), Box<EvalAltResult>> {
    // Get command-line arguments
    let args: Vec<String> = env::args().collect();

    // Check if a file path was provided
    if args.len() < 2 {
        eprintln!("Usage: rhai-runner <path/to/script.rhai>");
        process::exit(1);
    }

    let file_path = &args[1];
    println!("Running script: {}", file_path);

    // Create a new Rhai engine
    let mut engine = Engine::new();

    let random = RandomPackage::new();

    // Load the package into the `Engine`
    random.register_into_engine(&mut engine);

    // Read the script content from the file
    let script = match fs::read_to_string(file_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", file_path, e);
            process::exit(1);
        }
    };

    // Evaluate the script
    match engine.eval::<rhai::Dynamic>(&script) {
        Ok(result) => {
            println!("Script result: {}", result);
        }
        Err(e) => {
            eprintln!("Error evaluating script: {}", e);
            // Return the error to be printed with full details
            return Err(Box::new(*e));
        }
    }

    Ok(())
}
