@0xd90f2a231ad288f1;

struct MindInput {
    state @0: BlobState;
    context @1: BlobContext;
    seed @2: UInt64;
}

struct BlobState {
    energy @0: UInt32;
    minEnergy @1: UInt32;
    marker @2: UInt32;
    loaded @3: Bool;
    age @4: UInt32;
    memory @5: Data;
    messageQueue @6: List(Data);
}

struct BlobContext {
    elevation @0: SignedNeighborhood;
    energy @1: UnsignedNeighborhood;
    pheromone @2: UnsignedOptionNeighborhood;
    markers @3: UnsignedOptionNeighborhood;
}

struct SignedNeighborhood {
    center @0: Int32;
    north @1: Int32;
    south @2: Int32;
    east @3: Int32;
    west @4: Int32;
    northEast @5: Int32;
    northWest @6: Int32;
    southEast @7: Int32;
    southWest @8: Int32;
}

struct UnsignedNeighborhood {
    center @0: UInt32;
    north @1: UInt32;
    south @2: UInt32;
    east @3: UInt32;
    west @4: UInt32;
    northEast @5: UInt32;
    northWest @6: UInt32;
    southEast @7: UInt32;
    southWest @8: UInt32;
}
struct UnsignedOptionNeighborhood {
    center @0: UInt32Option;
    north @1: UInt32Option;
    south @2: UInt32Option;
    east @3: UInt32Option;
    west @4: UInt32Option;
    northEast @5: UInt32Option;
    northWest @6: UInt32Option;
    southEast @7: UInt32Option;
    southWest @8: UInt32Option;
}

struct UInt32Option {
    union {
        none @0: Void;
        some @1: UInt32;
    }
}