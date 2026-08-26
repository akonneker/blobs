@0x9eb0ecf3e36f7d41;

# Reference-native, cell-local Mind ABI. Slot indices and offsets are relative
# to the observing cell. This schema intentionally contains no cell/team ID,
# absolute coordinate, global time, population count, or random seed.

struct ReferenceMindInput {
    selfState @0 : ReferenceSelfState;
    slots @1 : List(LocalObservation);
    actionSpace @2 : ActionSpace;
    privateMemory @3 : Data;
    randomness @4 : Data;
    # The cell's own tile is not a target slot. Keeping it separate prevents
    # Consume observations from silently changing relative-action semantics.
    currentTile @5 : CurrentTileObservation;
}

struct CurrentTileObservation {
    elevation @0 : Int16;
    plantEnergy @1 : UInt64;
    looseEnergy @2 : UInt64;
    diffuseEnergy @3 : UInt64;
    plantCapacity @4 : UInt64;
    # Plant energy produced per simulation time unit when local diffuse energy
    # and capacity are available.
    plantGrowthRate @5 : UInt64;
    signalEnergy @6 : List(UInt64);
}

struct ReferenceSelfState {
    coreMass @0 : UInt64;
    assimilatedEnergy @1 : UInt64;
    gutEnergy @2 : UInt64;
    carriedMaterialMass @3 : UInt64;
    marker @4 : UInt32;
    guarded @5 : Bool;
    lastOutcome @6 : OptionalOutcome;
    metabolismRemainder @7 : UInt64;
}

struct LocalObservation {
    slot @0 : UInt8;
    dx @1 : Int8;
    dy @2 : Int8;
    distanceCostQ10 @3 : UInt16;
    reachable @4 : Bool;
    # Presence bits distinguish a hidden value from a visible zero without
    # allocating a pointer-backed optional object for every scalar. Unknown
    # bits are rejected. Neighbor detail bits require neighborPresent.
    visibility @5 : UInt16;
    elevation @6 : Int16;
    plantEnergy @7 : UInt64;
    plantCapacity @8 : UInt64;
    plantGrowthRate @9 : UInt64;
    looseEnergy @10 : UInt64;
    diffuseEnergy @11 : UInt64;
    signalEnergy0 @12 : UInt64;
    signalEnergy1 @13 : UInt64;
    signalEnergy2 @14 : UInt64;
    signalEnergy3 @15 : UInt64;
    neighborMarker @16 : UInt32;
    neighborApparentMassBucket @17 : UInt8;
    neighborActivity @18 : Activity;
    neighborProgress @19 : Progress;
}

struct ActionSpace {
    waitEnabled @0 : Bool;
    guardEnabled @1 : Bool;
    consumeEnabled @2 : Bool;
    moveTargets @3 : UInt32;
    attackTargets @4 : UInt32;
    splitTargets @5 : UInt32;
    regurgitateTargets @6 : UInt32;
    effortMask @7 : UInt8;
    maxConsumeAmount @8 : UInt64;
    gutCapacity @9 : UInt64;
    maxPrivateMemoryBytes @10 : UInt32;
    minimumSurvivalEnergy @11 : UInt64;
    childCoreMass @12 : UInt64;
    metabolismRateNumerator @13 : UInt64;
    metabolismRateDenominator @14 : UInt64;
    excavateEnabled @15 : Bool;
    depositTerrainEnabled @16 : Bool;
    signalEnabled @17 : Bool;
    terrainMassPerElevation @18 : UInt64;
    signalEmissionCost @19 : UInt64;
    effortCostNumerators @20 : List(UInt32);
    effortCostDenominators @21 : List(UInt32);
    moveEffortBase @22 : UInt64;
    moveMassUnitsPerEffort @23 : UInt64;
    attackEffortBase @24 : UInt64;
    guardEffortBase @25 : UInt64;
    consumeEffortBase @26 : UInt64;
    splitEffortBase @27 : UInt64;
    regurgitateEffortBase @28 : UInt64;
    excavateEffortBase @29 : UInt64;
    depositTerrainEffortBase @30 : UInt64;
}

struct OptionalOutcome {
    union {
        none @0 : Void;
        some @1 : Outcome;
    }
}

struct OptionalRejectReason {
    union {
        none @0 : Void;
        some @1 : RejectReason;
    }
}

enum Activity {
    ready @0;
    moving @1;
    attackWindup @2;
    guarding @3;
    feeding @4;
    splitting @5;
    manipulatingTerrain @6;
    otherBusy @7;
}

enum Progress {
    early @0;
    middle @1;
    late @2;
}

enum OutcomeStatus {
    success @0;
    frustrated @1;
    contested @2;
    interrupted @3;
    rejected @4;
}

enum RejectReason {
    invalidSlot @0;
    actionNotAllowedInSlot @1;
    targetOutsideWorld @2;
    targetsSelf @3;
    zeroPayload @4;
    insufficientGutEnergy @5;
    childAllocationTooSmall @6;
    privateMemoryTooLarge @7;
    insufficientEnergy @8;
    insufficientMaterial @9;
    terrainLimit @10;
    arithmeticOverflow @11;
}

struct Outcome {
    status @0 : OutcomeStatus;
    rejectedReason @1 : OptionalRejectReason;
}

enum EffortTier {
    gentle @0;
    standard @1;
    burst @2;
}

# Complete output of one pristine Mind invocation. Persistent state crosses
# the isolation boundary only through nextPrivateMemory.
struct ReferenceMindDecision {
    action @0 : ReferenceAction;
    nextPrivateMemory @1 : Data;
    signal @2 : OptionalSignalEmission;
    # When true, nextPrivateMemory must be empty and the cell's existing
    # private bytes are retained without a round trip through the Mind.
    retainPrivateMemory @3 : Bool;
}

struct OptionalSignalEmission {
    union {
        none @0 : Void;
        some @1 : SignalEmission;
    }
}

struct SignalEmission {
    channel @0 : UInt8;
    amount @1 : UInt64;
}

struct ReferenceAction {
    union {
        wait @0 : Void;
        move @1 : TargetedEffort;
        attack @2 : AttackData;
        guard @3 : EffortTier;
        consume @4 : UInt64;
        split @5 : SplitData;
        regurgitate @6 : RegurgitateData;
        excavate @7 : Void;
        depositTerrain @8 : Void;
        signal @9 : SignalVector;
    }
}

struct SignalVector {
    amount0 @0 : UInt64;
    amount1 @1 : UInt64;
    amount2 @2 : UInt64;
    amount3 @3 : UInt64;
}

struct TargetedEffort {
    targetSlot @0 : UInt8;
    effort @1 : EffortTier;
}

struct AttackData {
    targetSlot @0 : UInt8;
    effort @1 : EffortTier;
    payload @2 : UInt64;
}

struct SplitData {
    targetSlot @0 : UInt8;
    childAllocation @1 : UInt64;
    marker @2 : UInt32;
    privateMemory @3 : Data;
}

struct RegurgitateData {
    targetSlot @0 : UInt8;
    amount @1 : UInt64;
}
