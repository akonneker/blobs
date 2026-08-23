#![recursion_limit = "512"]
//! CLI entrypoint for blob_rl training.

use blob_rl::config::TrainingConfig;
use blob_rl::training::train;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "blob_rl_train",
    about = "Train blob minds via reinforcement learning"
)]
struct Args {
    /// Path to TOML config file
    #[arg(short, long)]
    config: Option<String>,

    /// Number of parallel environments
    #[arg(long)]
    num_envs: Option<usize>,

    /// Total training timesteps
    #[arg(long)]
    total_timesteps: Option<u64>,

    /// Rollout length per update
    #[arg(long)]
    rollout_length: Option<u64>,

    /// Learning rate
    #[arg(long)]
    lr: Option<f64>,

    /// Load a pre-trained model to continue training from
    #[arg(long)]
    load_model: Option<String>,

    /// Save the trained model to this path when done
    #[arg(long)]
    save_model: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Load config from file or use defaults
    let mut config = if let Some(path) = &args.config {
        TrainingConfig::from_file(path).expect("Failed to load config")
    } else {
        TrainingConfig::default()
    };

    // Apply CLI overrides
    if let Some(v) = args.num_envs {
        config.num_envs = v;
    }
    if let Some(v) = args.total_timesteps {
        config.total_timesteps = v;
    }
    if let Some(v) = args.rollout_length {
        config.rollout_length = v;
    }
    if let Some(v) = args.lr {
        config.ppo.learning_rate = v;
    }

    println!("Config: {:?}", config);

    let load_path = args.load_model.as_deref();
    let save_path = args.save_model.as_deref();

    // Run training with WGPU backend
    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        type MyBackend = Autodiff<Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        train::<MyBackend>(config, device, load_path, save_path);
    }

    #[cfg(not(feature = "wgpu"))]
    {
        use burn::backend::{Autodiff, NdArray};
        type MyBackend = Autodiff<NdArray>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        train::<MyBackend>(config, device, load_path, save_path);
    }
}
