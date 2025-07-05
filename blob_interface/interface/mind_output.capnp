@0xd0fda5351ba98b3e;

struct MindOutput {
    action @0: Action;
    memory @1: Data;
}

struct Action {
    union {
        sendMessage @0 : Message;
        defend @1 : Void;
        attack @2 : Direction;
        liftTerrain @3 : Void;
        dumpTerrain @4 : Void;
        setPheromone @5 : UInt32;
        move @6 : Direction;
        split @7 : SplitData;
        eat @8 : Void;
        doNothing @9 : Void;
    }
}

enum Direction {
    north @0;
    south @1;
    east @2;
    west @3;
    northEast @4;
    northWest @5;
    southEast @6;
    southWest @7;
}

struct Message {
    direction @0 : Direction;
    contents @1 : Data;
}

struct SplitData {
    direction @0 : Direction;
    energy @1: UInt32;
    marker @2: UInt32;
    memory @3: Data;
}