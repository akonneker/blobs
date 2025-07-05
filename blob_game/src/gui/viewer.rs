use crate::game::Game;
use egui::{ColorImage, TextureHandle, Scene, Rect};
use eframe::egui;
use std::error::Error;
use image::RgbImage;
use std::collections::HashMap;
use blob_interface::types::TeamId;

/// Launch the GUI viewer for the game
pub fn run_viewer(game: Game, verbose: bool) -> Result<(), Box<dyn Error>> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Blob Game Viewer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Blob Game Viewer",
        options,
        Box::new(move |cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            
            Ok(Box::new(GameViewer::new(game, verbose)))
        }),
    ).map_err(|e| Box::new(e) as Box<dyn Error>)
}

struct GameViewer {
    game: Game,
    verbose: bool,
    paused: bool,
    step_size: u64,
    auto_step: bool,
    scale: u32,
    board_texture: Option<TextureHandle>,
    last_update: std::time::Instant,
    update_interval: std::time::Duration,
    scene_rect: Rect, // Track the scene view area for zoom/pan
    reset_view_requested: bool, // Flag to request view reset from control panel
    game_ended: bool, // Track if the game has ended due to team elimination
    end_summary: String, // Store the game end summary
}

impl GameViewer {
    fn new(game: Game, verbose: bool) -> Self {
        Self {
            game,
            verbose,
            paused: true,
            step_size: 1,
            auto_step: true,
            scale: 8,
            board_texture: None,
            last_update: std::time::Instant::now(),
            update_interval: std::time::Duration::from_millis(50), // 20 FPS default
            scene_rect: Rect::ZERO, // egui::Scene will initialize this to something valid
            reset_view_requested: false,
            game_ended: false,
            end_summary: String::new(),
        }
    }
    
    fn get_team_color(&self, team_id: TeamId) -> egui::Color32 {
        // Same color palette as used in generate_board_image
        let team_colors = [
            [255, 100, 100], // Red
            [100, 100, 255], // Blue  
            [255, 255, 100], // Yellow
            [255, 100, 255], // Magenta
            [100, 255, 255], // Cyan
            [255, 165, 0],   // Orange
            [128, 0, 128],   // Purple
            [0, 128, 0],     // Dark Green
            [165, 42, 42],   // Brown
            [255, 20, 147],  // Deep Pink
        ];
        
        let team_index = team_id.0 % team_colors.len();
        let color = team_colors[team_index];
        egui::Color32::from_rgb(color[0], color[1], color[2])
    }
    
    fn update_board_texture(&mut self, ctx: &egui::Context) {
        if let Ok(rgb_image) = self.game.generate_board_image(self.scale, None, false) {
            let color_image = self.rgb_image_to_color_image(&rgb_image);
            
            // Always create a new texture since TextureHandle doesn't support mutation
            // Use nearest neighbor interpolation for crisp pixel rendering
            let texture_options = egui::TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Nearest,
                ..Default::default()
            };
            
            self.board_texture = Some(ctx.load_texture(
                "board",
                color_image,
                texture_options,
            ));
        }
    }
    
    fn rgb_image_to_color_image(&self, rgb_image: &RgbImage) -> ColorImage {
        let (width, height) = rgb_image.dimensions();
        let pixels: Vec<egui::Color32> = rgb_image
            .pixels()
            .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
            .collect();
        
        ColorImage {
            size: [width as usize, height as usize],
            pixels,
        }
    }
    
    fn step_game(&mut self) -> Result<(), extism::Error> {
        if self.game_ended {
            return Ok(()); // Don't step if game has ended
        }
        
        for _ in 0..self.step_size {
            if self.game.iteration >= self.game.max_iterations {
                break;
            }
            self.game.tick(self.verbose)?;
            
            // Check for team elimination after each tick
            if let Some(surviving_teams) = self.game.check_team_elimination() {
                self.end_summary = self.game.generate_game_summary(&surviving_teams);
                self.game_ended = true;
                self.auto_step = false; // Stop auto-stepping
                println!("{}", self.end_summary);
                break;
            }
        }
        Ok(())
    }
}

impl eframe::App for GameViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Auto-step logic
        if self.auto_step && !self.paused && self.last_update.elapsed() >= self.update_interval {
            if let Err(e) = self.step_game() {
                eprintln!("Error stepping game: {}", e);
                self.auto_step = false;
            }
            self.update_board_texture(ctx);
            self.last_update = std::time::Instant::now();
        }
        
        // Control panel
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                
                // Game info
                ui.label(format!("Iteration: {}/{}", self.game.iteration, self.game.max_iterations));
                ui.label(format!("Cells alive: {}", self.game.cells.len()));
                ui.label(format!("Teams: {}", self.game.teams.len()));
                
                // Game status
                if self.game_ended {
                    ui.colored_label(egui::Color32::RED, "🎮 GAME ENDED");
                    if ui.button("📊 Show Summary").clicked() {
                        // Print summary to console again
                        println!("{}", self.end_summary);
                    }
                } else if self.game.iteration >= self.game.max_iterations {
                    ui.colored_label(egui::Color32::YELLOW, "⏰ MAX ITERATIONS REACHED");
                } else {
                    ui.colored_label(egui::Color32::GREEN, "🔄 RUNNING");
                }
                
                ui.separator();
                
                // Controls
                ui.add_enabled_ui(!self.game_ended, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                            self.paused = !self.paused;
                        }
                        
                        if ui.button("⏭ Step").clicked() {
                            if let Err(e) = self.step_game() {
                                eprintln!("Error stepping game: {}", e);
                            }
                            self.update_board_texture(ctx);
                        }
                    });
                    
                    ui.checkbox(&mut self.auto_step, "Auto-step");
                });
                
                ui.add(egui::Slider::new(&mut self.step_size, 1..=100).text("Step size"));
                
                // Update interval for auto-step
                let mut interval_ms = self.update_interval.as_millis() as u64;
                ui.add(egui::Slider::new(&mut interval_ms, 50..=2000).text("Update interval (ms)"));
                self.update_interval = std::time::Duration::from_millis(interval_ms);
                
                ui.separator();
                
                // Display controls
                if ui.button("🔄 Refresh Image").clicked() {
                    self.update_board_texture(ctx);
                }
                
                // Reset view button moved here
                if ui.button("🏠 Reset View").clicked() {
                    self.reset_view_requested = true;
                }
                
                ui.separator();
                
                // Zoom/Pan info
                ui.label("Zoom/Pan:");
                ui.label("• Scroll to zoom");
                ui.label("• Cmd/Ctrl+Scroll to zoom");
                ui.label("• Drag to pan");
                ui.label("• Double-click to reset view");
                ui.label(format!("Scene rect: {:#?}", self.scene_rect));
                
                ui.separator();
                
                // Export options
                if ui.button("💾 Save Image").clicked() {
                    let filename = format!("game_state_{:06}.png", self.game.iteration);
                    if let Err(e) = self.game.generate_board_image(self.scale, Some(&filename), true) {
                        eprintln!("Failed to save image: {}", e);
                    } else {
                        println!("Saved image: {}", filename);
                    }
                }
                
                ui.separator();
                
                // State management
                ui.label("State Management:");
                
                if ui.button("💾 Save State").clicked() {
                    let filename = format!("game_state_{:06}.gz", self.game.iteration);
                    if let Err(e) = self.game.save_state(&filename, true) {
                        eprintln!("Failed to save state: {}", e);
                    } else {
                        println!("Saved state: {}", filename);
                    }
                }
                
                if ui.button("💾 Save State (Uncompressed)").clicked() {
                    let filename = format!("game_state_{:06}.bin", self.game.iteration);
                    if let Err(e) = self.game.save_state(&filename, false) {
                        eprintln!("Failed to save state: {}", e);
                    } else {
                        println!("Saved state: {}", filename);
                    }
                }
                
                ui.separator();
                
                // Teams section
                egui::CollapsingHeader::new(format!("Teams ({})", self.game.teams.len()))
                    .default_open(true)
                    .show(ui, |ui| {
                        // Group cells by team and include all teams (even eliminated ones)
                        let mut teams_cells: HashMap<TeamId, Vec<_>> = HashMap::new();
                        
                        // Initialize all teams with empty vectors
                        for team_id in self.game.teams.keys() {
                            teams_cells.insert(*team_id, Vec::new());
                        }
                        
                        // Fill in the teams that have cells
                        for (cell_id, cell) in &self.game.cells {
                            teams_cells.entry(cell.team_id).or_insert_with(Vec::new).push((cell_id, cell));
                        }
                        
                        // Sort teams by ID for consistent display order
                        let mut sorted_teams: Vec<_> = teams_cells.iter().collect();
                        sorted_teams.sort_by_key(|(team_id, _)| team_id.0);
                        
                        for (team_id, cells) in sorted_teams {
                            let team_color = self.get_team_color(*team_id);
                            let team_status = if cells.is_empty() { " [ELIMINATED]" } else { "" };
                            let header_color = if cells.is_empty() { egui::Color32::GRAY } else { team_color };
                            
                            egui::CollapsingHeader::new(egui::RichText::new(format!("Team {} ({} cells){}", team_id.0, cells.len(), team_status)).color(header_color))
                                .default_open(false)
                                .show(ui, |ui| {
                                    // Team statistics
                                    let total_energy: u32 = cells.iter().map(|(_, cell)| cell.energy).sum();
                                    let avg_energy = if !cells.is_empty() { total_energy / cells.len() as u32 } else { 0 };
                                    let avg_age: f32 = if !cells.is_empty() { 
                                        cells.iter().map(|(_, cell)| cell.age as f32).sum::<f32>() / cells.len() as f32 
                                    } else { 0.0 };
                                    
                                    ui.label(format!("Total Energy: {}", total_energy));
                                    ui.label(format!("Average Energy: {}", avg_energy));
                                    ui.label(format!("Average Age: {:.1}", avg_age));
                                    
                                    ui.separator();
                                    
                                    // Cell table
                                    egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                                        egui::Grid::new(format!("team_{}_cells", team_id.0))
                                            .striped(true)
                                            .show(ui, |ui| {
                                                // Table headers
                                                ui.strong("ID");
                                                ui.strong("Position");
                                                ui.strong("Energy");
                                                ui.strong("Age");
                                                ui.strong("Marker");
                                                ui.strong("Status");
                                                ui.end_row();
                                                
                                                // Sort cells by ID for consistent display
                                                let mut sorted_cells = cells.clone();
                                                sorted_cells.sort_by_key(|(cell_id, _)| cell_id.0);
                                                
                                                for (cell_id, cell) in sorted_cells {
                                                    ui.label(format!("{}", cell_id.0));
                                                    
                                                    // Get position from coordinate map
                                                    if let Some(pos) = self.game.inv_coordinate_map.get(cell_id) {
                                                        ui.label(format!("({}, {})", pos.x, pos.y));
                                                    } else {
                                                        ui.label("Unknown");
                                                    }
                                                    
                                                    ui.label(format!("{}", cell.energy));
                                                    ui.label(format!("{}", cell.age));
                                                    ui.label(format!("{}", cell.marker));
                                                    
                                                    // Status indicators
                                                    let mut status_parts: Vec<String> = Vec::new();
                                                    if cell.defending {
                                                        status_parts.push("Defending".to_string());
                                                    }
                                                    if cell.loaded {
                                                        status_parts.push("Loaded".to_string());
                                                    }
                                                    if !cell.message_queue.is_empty() {
                                                        status_parts.push(format!("{}msg", cell.message_queue.len()));
                                                    }
                                                    let status = if status_parts.is_empty() {
                                                        "Normal".to_string()
                                                    } else {
                                                        status_parts.join(", ")
                                                    };
                                                    ui.label(status);
                                                    
                                                    ui.end_row();
                                                }
                                            });
                                    });
                                });
                        }
                        
                        if teams_cells.is_empty() {
                            ui.label("No teams found");
                        }
                    });
            });
        });
        
        // Main game view with Scene for zoom/pan
        egui::CentralPanel::default().show(ctx, |ui| {
            
            if self.board_texture.is_none() {
                self.update_board_texture(ctx);
            }
            
            // Create a frame for the scene
            egui::Frame::group(ui.style())
                .inner_margin(0.0)
                .show(ui, |ui| {
                    let scene = Scene::new()
                        .zoom_range(0.1..=10.0); // Allow zooming from 10% to 1000%
                    
                    let mut inner_rect = Rect::NAN;
                    
                    let response = scene
                        .show(ui, &mut self.scene_rect, |ui| {
                            if let Some(texture) = &self.board_texture {
                                let texture_size = texture.size_vec2();
                                
                                // Place the game board image
                                ui.image((texture.id(), texture_size));
                                
                                // Track the area for reset functionality
                                inner_rect = ui.min_rect();
                            } else {
                                ui.label("Loading game board...");
                                inner_rect = ui.min_rect();
                            }
                        })
                        .response;
                    
                    // Reset view if requested from control panel or double-clicked
                    if self.reset_view_requested || response.double_clicked() {
                        self.scene_rect = inner_rect;
                        self.reset_view_requested = false; // Clear the flag
                    }
                });
        });
        
        // Request repaint for auto-step
        if self.auto_step && !self.paused {
            ctx.request_repaint();
        }
    }
}