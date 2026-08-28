#![recursion_limit = "512"]
//! CLI entrypoint for blob_rl training.

use blob_rl::config::{InitialPolicyConfig, InitialPolicyQualificationConfig, TrainingConfig};
use blob_rl::training::train;
use clap::Parser;
use std::fs::OpenOptions;
use std::path::PathBuf;

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

    /// Hard ceiling on sampled cell actions
    #[arg(long)]
    total_timesteps: Option<u64>,

    /// Required cumulative authoritative simulation quanta in every environment
    #[arg(long)]
    total_simulation_quanta_per_env: Option<u64>,

    /// Rollout length per update
    #[arg(long)]
    rollout_length: Option<u64>,

    /// Learning rate
    #[arg(long)]
    lr: Option<f64>,

    /// PPO samples per optimizer step; GPU runs generally need a much larger
    /// batch than the portable CPU default to amortize dispatch overhead
    #[arg(long)]
    minibatch_size: Option<usize>,

    /// Reproducible environment and policy-sampling seed
    #[arg(long)]
    seed: Option<u64>,

    /// Updates between held-out evaluations; 0 disables evaluation
    #[arg(long)]
    eval_interval: Option<usize>,

    /// Number of fixed held-out episodes per evaluation
    #[arg(long)]
    eval_episodes: Option<usize>,

    /// First seed in the fixed held-out evaluation suite
    #[arg(long)]
    evaluation_seed: Option<u64>,

    /// Updates between periodic immutable checkpoints; 0 disables them
    #[arg(long)]
    checkpoint_interval: Option<usize>,

    /// Directory for metrics and immutable checkpoints
    #[arg(long)]
    checkpoint_dir: Option<String>,

    /// Immutable behavior-cloning artifact directory used for a fresh run.
    #[arg(long, conflicts_with = "load_model")]
    initial_policy: Option<String>,

    /// Expected SHA-256 of the behavior clone's metadata bytes.
    #[arg(long)]
    initial_policy_artifact_sha256: Option<String>,

    /// Passing feeding-evaluation JSON that qualifies the behavior clone.
    #[arg(long)]
    initial_policy_qualification: Option<String>,

    /// Expected semantic hash embedded in the feeding qualification.
    #[arg(long)]
    initial_policy_qualification_hash: Option<String>,

    /// Load a pre-trained model to continue training from
    #[arg(long, conflicts_with = "resume")]
    load_model: Option<String>,

    /// Resume all training state from an immutable update-boundary checkpoint
    #[arg(long, conflicts_with = "load_model")]
    resume: Option<String>,

    /// Save the trained model to this path when done
    #[arg(long)]
    save_model: Option<String>,

    /// Executor-owned lease held for the complete trainer lifetime. This
    /// prevents an orphaned child and a replacement sweep executor from
    /// training the same immutable run concurrently.
    #[arg(long, hide = true)]
    run_lock: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let _run_lease = args.run_lock.as_ref().map(|path| {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .unwrap_or_else(|error| panic!("failed to open run lease {}: {error}", path.display()));
        file.try_lock().unwrap_or_else(|error| {
            panic!(
                "another trainer holds the run lease {}: {error}",
                path.display()
            )
        });
        file
    });

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
    if let Some(v) = args.total_simulation_quanta_per_env {
        config.total_simulation_quanta_per_env = Some(v);
    }
    if let Some(v) = args.rollout_length {
        config.rollout_length = v;
    }
    if let Some(v) = args.lr {
        config.ppo.learning_rate = v;
    }
    if let Some(v) = args.minibatch_size {
        config.ppo.minibatch_size = v;
    }
    if let Some(v) = args.seed {
        config.seed = v;
    }
    if let Some(v) = args.eval_interval {
        config.eval_interval = v;
    }
    if let Some(v) = args.eval_episodes {
        config.eval_episodes = v;
    }
    if let Some(v) = args.evaluation_seed {
        config.evaluation_seed = v;
    }
    if let Some(v) = args.checkpoint_interval {
        config.checkpoint_interval = v;
    }
    if let Some(v) = args.checkpoint_dir {
        config.checkpoint_dir = v;
    }
    match (
        args.initial_policy,
        args.initial_policy_artifact_sha256,
        args.initial_policy_qualification,
        args.initial_policy_qualification_hash,
    ) {
        (None, None, None, None) => {}
        (Some(directory), Some(artifact_sha256), Some(path), Some(artifact_hash)) => {
            config.initial_policy = Some(InitialPolicyConfig {
                directory,
                artifact_sha256,
                qualification: Some(InitialPolicyQualificationConfig {
                    path,
                    artifact_hash,
                }),
            });
        }
        _ => panic!(
            "--initial-policy requires its artifact SHA-256, qualification path, and qualification hash"
        ),
    }

    config.validate().expect("Invalid training configuration");

    println!("Config: {:?}", config);
    let load_path = args.load_model.as_deref();
    let resume_path = args.resume.as_deref().map(std::path::Path::new);
    let save_path = args.save_model.as_deref();

    // Run training with WGPU backend
    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        type MyBackend = Autodiff<Wgpu>;
        println!(
            "Training backend: {}",
            blob_rl::artifact::training_backend_id::<MyBackend>()
        );
        let device = burn::backend::wgpu::WgpuDevice::default();
        train::<MyBackend>(config, device, load_path, resume_path, save_path);
    }

    #[cfg(not(feature = "wgpu"))]
    {
        use burn::backend::{Autodiff, NdArray};
        type MyBackend = Autodiff<NdArray>;
        println!(
            "Training backend: {}",
            blob_rl::artifact::training_backend_id::<MyBackend>()
        );
        let device = burn::backend::ndarray::NdArrayDevice::default();
        train::<MyBackend>(config, device, load_path, resume_path, save_path);
    }
}
