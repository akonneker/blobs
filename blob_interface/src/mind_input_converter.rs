use crate::cell::{BlobState, Cell, CellContext};
use crate::mind_input_capnp::{mind_input, signed_neighborhood, u_int32_option,
    unsigned_neighborhood, unsigned_option_neighborhood,
};
use capnp::serialize;
use crate::types::CellMessage;

pub fn cell_to_mind_input_capnp(cell: &Cell, context: &CellContext, seed: u64) -> capnp::Result<Vec<u8>> {
    let mut message = ::capnp::message::Builder::new_default();
    let mut mind_input_builder = message.init_root::<mind_input::Builder>();

    mind_input_builder.set_seed(seed);

    // Build BlobState
    let mut state_builder = mind_input_builder.reborrow().init_state();
    state_builder.set_energy(cell.energy);
    state_builder.set_min_energy(cell.min_energy);
    state_builder.set_max_energy(cell.max_energy);
    state_builder.set_marker(cell.marker);
    state_builder.set_loaded(cell.loaded);
    state_builder.set_age(cell.age);
    state_builder.set_memory(&cell.memory);
    let mut message_queue_builder = state_builder.init_message_queue(cell.message_queue.len() as u32);
    for (i, msg) in cell.message_queue.iter().enumerate() {
        message_queue_builder.reborrow().set(i as u32, &msg.data);
    }

    // Build BlobContext
    let mut context_builder = mind_input_builder.reborrow().init_context();
    
    // Elevation
    let mut elevation_builder = context_builder.reborrow().init_elevation();
    fill_signed_neighborhood(&mut elevation_builder, &context.elevation);

    // Energy
    let mut energy_builder = context_builder.reborrow().init_energy();
    fill_unsigned_neighborhood(&mut energy_builder, &context.energy);

    // Pheromone
    let mut pheromone_builder = context_builder.reborrow().init_pheromone();
    fill_unsigned_option_neighborhood(&mut pheromone_builder, &context.pheromone);

    // Markers
    // Note: context.markers has 8 elements, neighborhood has 9.
    // Assuming no center marker is provided from context.
    let mut markers_builder = context_builder.reborrow().init_markers();
    fill_unsigned_option_neighborhood_8(&mut markers_builder, &context.markers);

    let mut output = Vec::new();
    serialize::write_message(&mut output, &message)?;
    Ok(output)
}

fn fill_signed_neighborhood(builder: &mut signed_neighborhood::Builder, data: &[i32; 9]) {
    builder.set_north_west(data[0]);
    builder.set_north(data[1]);
    builder.set_north_east(data[2]);
    builder.set_west(data[3]);
    builder.set_east(data[4]);
    builder.set_south_west(data[5]);
    builder.set_south(data[6]);
    builder.set_south_east(data[7]);
    builder.set_center(data[8]);
}

fn fill_unsigned_neighborhood(builder: &mut unsigned_neighborhood::Builder, data: &[u32; 9]) {
    builder.set_north_west(data[0]);
    builder.set_north(data[1]);
    builder.set_north_east(data[2]);
    builder.set_west(data[3]);
    builder.set_east(data[4]);
    builder.set_south_west(data[5]);
    builder.set_south(data[6]);
    builder.set_south_east(data[7]);
    builder.set_center(data[8]);
}

fn set_option(mut builder: u_int32_option::Builder, value: Option<u32>) {
    match value {
        Some(v) => builder.set_some(v),
        None => builder.set_none(()),
    }
}

fn fill_unsigned_option_neighborhood(
    builder: &mut unsigned_option_neighborhood::Builder,
    data: &[Option<u32>; 9],
) {
    set_option(builder.reborrow().init_north_west(), data[0]);
    set_option(builder.reborrow().init_north(), data[1]);
    set_option(builder.reborrow().init_north_east(), data[2]);
    set_option(builder.reborrow().init_west(), data[3]);
    set_option(builder.reborrow().init_east(), data[4]);
    set_option(builder.reborrow().init_south_west(), data[5]);
    set_option(builder.reborrow().init_south(), data[6]);
    set_option(builder.reborrow().init_south_east(), data[7]);
    set_option(builder.reborrow().init_center(), data[8]);
}

fn fill_unsigned_option_neighborhood_8(
    builder: &mut unsigned_option_neighborhood::Builder,
    data: &[Option<u32>; 8],
) {
    set_option(builder.reborrow().init_north_west(), data[0]);
    set_option(builder.reborrow().init_north(), data[1]);
    set_option(builder.reborrow().init_north_east(), data[2]);
    set_option(builder.reborrow().init_west(), data[3]);
    set_option(builder.reborrow().init_east(), data[4]);
    set_option(builder.reborrow().init_south_west(), data[5]);
    set_option(builder.reborrow().init_south(), data[6]);
    set_option(builder.reborrow().init_south_east(), data[7]);
    builder.reborrow().init_center().set_none(());
}

pub fn capnp_to_mind_input(data: &[u8]) -> capnp::Result<(BlobState, CellContext, u64)> {
    let message_reader =
        capnp::serialize::read_message(data, ::capnp::message::ReaderOptions::new())?;
    let mind_input_reader = message_reader.get_root::<mind_input::Reader>()?;

    // Read BlobState
    let state_reader = mind_input_reader.get_state()?;
    let memory_reader = state_reader.get_memory()?;
    let mut memory = [0u8; 2048];
    let len = std::cmp::min(memory_reader.len() as usize, 2048);
    memory[..len].copy_from_slice(&memory_reader[..len]);

    let message_queue_reader = state_reader.get_message_queue()?;
    let mut message_queue = Vec::with_capacity(message_queue_reader.len() as usize);
    for i in 0..message_queue_reader.len() {
        let msg_reader = message_queue_reader.get(i)?;
        let msg_data = msg_reader.to_vec();
        message_queue.push(CellMessage { data: msg_data });
    }

    let blob_state = BlobState {
        energy: state_reader.get_energy(),
        min_energy: state_reader.get_min_energy(),
        max_energy: state_reader.get_max_energy(),
        marker: state_reader.get_marker(),
        loaded: state_reader.get_loaded(),
        age: state_reader.get_age(),
        memory,
        message_queue,
    };

    let seed = mind_input_reader.get_seed();

    // Read CellContext
    let context_reader = mind_input_reader.get_context()?;
    let elevation = read_signed_neighborhood(&context_reader.get_elevation()?)?;
    let energy = read_unsigned_neighborhood(&context_reader.get_energy()?)?;
    let pheromone = read_unsigned_option_neighborhood(&context_reader.get_pheromone()?)?;
    let markers = read_unsigned_option_neighborhood_to_8(&context_reader.get_markers()?)?;

    let cell_context = CellContext {
        elevation,
        energy,
        pheromone,
        markers,
    };

    Ok((blob_state, cell_context, seed))
}

fn read_signed_neighborhood(reader: &signed_neighborhood::Reader) -> capnp::Result<[i32; 9]> {
    let mut data = [0; 9];
    data[0] = reader.get_north_west();
    data[1] = reader.get_north();
    data[2] = reader.get_north_east();
    data[3] = reader.get_west();
    data[4] = reader.get_east();
    data[5] = reader.get_south_west();
    data[6] = reader.get_south();
    data[7] = reader.get_south_east();
    data[8] = reader.get_center();
    Ok(data)
}

fn read_unsigned_neighborhood(reader: &unsigned_neighborhood::Reader) -> capnp::Result<[u32; 9]> {
    let mut data = [0; 9];
    data[0] = reader.get_north_west();
    data[1] = reader.get_north();
    data[2] = reader.get_north_east();
    data[3] = reader.get_west();
    data[4] = reader.get_east();
    data[5] = reader.get_south_west();
    data[6] = reader.get_south();
    data[7] = reader.get_south_east();
    data[8] = reader.get_center();
    Ok(data)
}

fn read_u_int32_option(reader: u_int32_option::Reader) -> capnp::Result<Option<u32>> {
    match reader.which()? {
        u_int32_option::Which::Some(val) => Ok(Some(val)),
        u_int32_option::Which::None(()) => Ok(None),
    }
}

fn read_unsigned_option_neighborhood(
    reader: &unsigned_option_neighborhood::Reader,
) -> capnp::Result<[Option<u32>; 9]> {
    let mut data: [Option<u32>; 9] = Default::default();
    data[0] = read_u_int32_option(reader.get_north_west()?)?;
    data[1] = read_u_int32_option(reader.get_north()?)?;
    data[2] = read_u_int32_option(reader.get_north_east()?)?;
    data[3] = read_u_int32_option(reader.get_west()?)?;
    data[4] = read_u_int32_option(reader.get_east()?)?;
    data[5] = read_u_int32_option(reader.get_south_west()?)?;
    data[6] = read_u_int32_option(reader.get_south()?)?;
    data[7] = read_u_int32_option(reader.get_south_east()?)?;
    data[8] = read_u_int32_option(reader.get_center()?)?;
    Ok(data)
}

fn read_unsigned_option_neighborhood_to_8(
    reader: &unsigned_option_neighborhood::Reader,
) -> capnp::Result<[Option<u32>; 8]> {
    let mut data: [Option<u32>; 8] = Default::default();
    data[0] = read_u_int32_option(reader.get_north_west()?)?;
    data[1] = read_u_int32_option(reader.get_north()?)?;
    data[2] = read_u_int32_option(reader.get_north_east()?)?;
    data[3] = read_u_int32_option(reader.get_west()?)?;
    data[4] = read_u_int32_option(reader.get_east()?)?;
    data[5] = read_u_int32_option(reader.get_south_west()?)?;
    data[6] = read_u_int32_option(reader.get_south()?)?;
    data[7] = read_u_int32_option(reader.get_south_east()?)?;
    Ok(data)
} 