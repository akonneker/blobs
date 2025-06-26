pub mod viewer;

use crate::game::Game;
use std::error::Error;

/// Launch the GUI for the game
pub fn launch_gui(game: Game, verbose: bool) -> Result<(), Box<dyn Error>> {
    println!("Launching GUI...");
    viewer::run_viewer(game, verbose)
} 