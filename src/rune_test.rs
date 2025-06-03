use rune::runtime::Vm;
use rune::{Context, Sources};
use rune::termcolor::{ColorChoice, StandardStream};
use std::sync::Arc;
use std::path::Path;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about = "Rune script runner")]
struct Args {
    /// Path to the Rune script file to execute
    #[arg(required = true)]
    script_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    let script_path = Path::new(&args.script_path);
    let context = Context::with_default_modules()?;
    let mut sources = Sources::new();

    if !script_path.exists() {
        eprintln!("Error: Script file '{}' not found.", script_path.display());
        return Err("Script file not found".into());
    }
    sources.insert(rune::Source::from_path(script_path)?)?;
    let mut diagnostics = rune::Diagnostics::new();

    let result = rune::prepare(&mut sources)
        .with_context(&context)
        .with_diagnostics(&mut diagnostics)
        .build();

    if !diagnostics.is_empty() {
        let mut writer = StandardStream::stderr(ColorChoice::Always);
        diagnostics.emit(&mut writer, &sources)?;
    }

    match result {
        Ok(unit) => {
            println!("Successfully compiled '{}' into a Rune Unit.", script_path.display());
            let unit = Arc::new(unit); // Often good to Arc the unit for sharing

            // --- Optional: Now you can use the Unit with a VM ---
            println!("\nAttempting to run the 'main' function from the compiled unit...");
            let mut vm = Vm::new(Arc::new(context.runtime()?), unit.clone());
            

            // Call the 'main' function (assuming it exists and takes no arguments)
            match vm.call(["main"], ()) {
                Ok(output) => {
                    let result_value: i64 = rune::from_value(output)?; // Assuming main returns an integer
                    println!("'main' function executed and returned: {}", result_value);
                }
                Err(e) => {
                    eprintln!("\nError executing 'main' function:");
                    let mut writer = StandardStream::stderr(ColorChoice::Always);
                    // Make sure sources is still available if the error needs to refer to it.
                    // For VmError, we pass the Vm's sources (which would be from the unit).
                    e.emit(&mut writer, &sources)?;
                    
                }
            }

            Ok(())
        }
        Err(e) => {
            // If `build()` failed, `e` will be a `CompileError`.
            // Diagnostics should have already been printed if they were generated.
            // `e` itself might not contain all diagnostic details directly if `diagnostics.emit` was used.
            eprintln!("\nFailed to compile '{}'.", script_path.display());
            // The error `e` (rune::compile::CompileError) can also be printed,
            // but `diagnostics.emit` usually gives more user-friendly output.
            // For example, just to show the error:
            // eprintln!("Compilation Error: {:?}", e);
            Err(Box::new(e)) // Convert to a generic error to satisfy main's return type
        }
    }
}