//! Object behaviours: names, coarse categories, and the fields worth reading per
//! behaviour (all names are `Object` fields of `fields.tsv`; a test checks they exist).

use std::collections::HashMap;
use std::sync::OnceLock;

use obrrt_core::layout::BEHAVIOURS;

use super::*;

/// Category of a behaviour by name.
pub fn categorize(name: &str) -> u8 {
    const COSMETIC: [&str; 14] = ["Particle", "Sparkle", "Mist", "Smoke", "Splash", "Bubble", "Puff", "Wave", "Dust", "Shadow", "Warp", "Spawn", "Menu", "Sound"];
    if COSMETIC.iter().any(|c| name.contains(c)) && !name.contains("Coin") && !name.contains("Star") {
        return CAT_NONE;
    }
    if name == "bhvMario" {
        return CAT_NONE;
    }
    if name.contains("Coin") && !name.contains("Sparkle") {
        return CAT_COIN;
    }
    if (name.contains("Star") && !name.contains("StarDoor") && !name.contains("UnlockDoor")) || name == "bhvGrandStar" {
        return CAT_STAR;
    }
    if name.contains("Pole") || name == "bhvTree" || name == "bhvTreeSnow" {
        return CAT_POLE;
    }
    if name.contains("Bobomb") && !name.contains("Buddy") && !name.contains("Explosion") {
        return CAT_BOMB;
    }
    if name.contains("Breakable") || name.contains("Cork") || name.contains("BreakBox") || name.contains("ExclamationBox") || name == "bhvJumpingBox" {
        return CAT_BREAKABLE;
    }
    const ENEMY: [&str; 30] = [
        "Goomba", "Koopa", "ChainChomp", "Thwomp", "Grindel", "Whomp", "Bully", "Piranha", "Boo", "MrI", "Chuckya", "HeaveHo", "Scuttlebug",
        "Amp", "Spindrift", "Snufit", "Swoop", "MadPiano", "Monty", "Skeeter", "FlyGuy", "Bub", "Sushi", "Bowser", "BulletBill", "Flame",
        "Fire", "Ukiki", "Mole", "Tox",
    ];
    if ENEMY.iter().any(|e| name.contains(e)) && !name.contains("Shell") && !name.contains("Cage") {
        return CAT_ENEMY;
    }
    const PLAT: [&str; 14] = [
        "Platform", "Elevator", "Lift", "Seesaw", "Tilting", "Bridge", "Rotating", "Moving", "Checkerboard", "Mesh", "Sliding", "Squarish",
        "Tower", "Rock",
    ];
    if PLAT.iter().any(|e| name.contains(e)) {
        return CAT_PLATFORM;
    }
    CAT_OTHER
}

/// Category of a behaviour script address (cached table over all behaviours).
pub fn cat_of(bhv: u32) -> u8 {
    static T: OnceLock<HashMap<u32, u8>> = OnceLock::new();
    *T.get_or_init(|| BEHAVIOURS.iter().map(|&(a, n)| (a, categorize(n))).collect()).get(&bhv).unwrap_or(&CAT_OTHER)
}

pub fn category_by_name(s: &str) -> Option<u8> {
    CAT_NAMES.iter().position(|n| *n == s).map(|i| i as u8)
}

pub fn default_radius(c: u8) -> f32 {
    match c {
        CAT_COIN => 100.0,
        CAT_STAR => 160.0,
        CAT_POLE => 80.0,
        CAT_ENEMY | CAT_BOMB => 100.0,
        CAT_PLATFORM => 300.0,
        _ => 100.0,
    }
}

/// Fields every object has and policies commonly want.
pub const COMMON: &[&str] = &[
    "oPosX", "oPosY", "oPosZ", "oVelX", "oVelY", "oVelZ", "oForwardVel", "oAction", "oSubAction", "oPrevAction", "oTimer", "oHealth",
    "oHeldState", "oInteractStatus", "oInteractType", "oIntangibleTimer", "oDistanceToMario", "oAngleToMario", "oMoveAngleYaw",
    "oFaceAngleYaw", "oAngleVelYaw", "oBehParams", "oBehParams2ndByte", "oHomeX", "oHomeY", "oHomeZ", "oFloorHeight", "oMoveFlags",
    "oDamageOrCoinValue", "oNumLootCoins", "hitboxRadius", "hitboxHeight", "hurtboxRadius", "hurtboxHeight",
];

const PLAT_KEYS: &[&str] = &["oAction", "oPosY", "oHomeY", "oVelY", "oForwardVel", "oAngleVelYaw", "oFaceAngleYaw", "oFaceAnglePitch", "oAngleVelPitch", "oTimer"];

/// Key fields per behaviour (beyond [`COMMON`]): the state a policy would plan around.
pub const KEYS: &[(&str, &[&str])] = &[
    ("bhvBobomb", &["oBobombFuseLit", "oBobombFuseTimer", "oBobombBlinkTimer", "oAction", "oHeldState", "oForwardVel"]),
    ("bhvKingBobomb", &["oAction", "oSubAction", "oTimer", "oHealth", "oHeldState", "oForwardVel", "oDistanceToMario"]),
    ("bhvBobombBuddy", &["oBobombBuddyRole", "oBobombBuddyHasTalkedToMario", "oBobombBuddyCannonStatus", "oAction"]),
    ("bhvChainChomp", &["oAction", "oChainChompDistToPivot", "oChainChompReleaseStatus", "oChainChompNumLunges", "oChainChompHitGate", "oChainChompRestrictedByChain", "oChainChompMaxDistFromPivotPerChainPart"]),
    ("bhvBowser", &["oAction", "oBowserStatus", "oBowserTimer", "oBowserCamAct", "oBowserDistToCentre", "oBowserAngleToCentre", "oBowserGrabbedStatus", "oBowserIsReacting", "oHealth", "oBowserTargetOpacity"]),
    ("bhvGoomba", &["oAction", "oGoombaSize", "oGoombaScale", "oGoombaWalkTimer", "oGoombaTargetYaw", "oGoombaTurningAwayFromWall", "oGoombaRelativeSpeed"]),
    ("bhvKoopa", &["oAction", "oKoopaAgility", "oKoopaMovementType", "oKoopaDistanceToMario", "oKoopaAngleToMario", "oKoopaTheQuickRaceIndex", "oKoopaCountdown"]),
    ("bhvKoopaRaceEndpoint", &["oKoopaRaceEndpointRaceBegun", "oKoopaRaceEndpointKoopaFinished", "oKoopaRaceEndpointRaceStatus", "oKoopaRaceEndpointRaceEnded"]),
    ("bhvThwomp", &["oAction", "oThwompRandomTimer", "oTimer", "oPosY", "oHomeY"]),
    ("bhvGrindel", &["oAction", "oTimer", "oPosY", "oHomeY"]),
    ("bhvWhompKingBoss", &["oAction", "oHealth", "oTimer", "oForwardVel", "oVelY"]),
    ("bhvSmallWhomp", &["oAction", "oHealth", "oTimer", "oForwardVel"]),
    ("bhvBoo", &["oBooDeathStatus", "oBooTargetOpacity", "oBooBaseScale", "oBooNegatedAggressiveness", "oAction", "oOpacity"]),
    ("bhvGhostHuntBoo", &["oBooDeathStatus", "oBooTargetOpacity", "oAction", "oOpacity"]),
    ("bhvSmallBully", &["oBullySubtype", "oBullyKBTimerAndMinionKOCounter", "oBullyMarioCollisionAngle", "oAction", "oForwardVel"]),
    ("bhvPiranhaPlant", &["oAction", "oPiranhaPlantScale", "oPiranhaPlantSleepMusicState"]),
    ("bhvFirePiranhaPlant", &["oAction", "oFirePiranhaPlantActive", "oFirePiranhaPlantScale", "oHealth"]),
    ("bhvMrI", &["oAction", "oMrISize", "oMrIUnk110"]),
    ("bhvChuckya", &["oAction", "oChuckyaUnkF8", "oChuckyaUnkFC", "oHeldState"]),
    ("bhvHeaveHo", &["oAction", "oHeaveHoUnkF4"]),
    ("bhvScuttlebug", &["oAction", "oScuttlebugUnkF4", "oScuttlebugUnkF8"]),
    ("bhvSnufit", &["oAction", "oSnufitRecoil", "oSnufitScale", "oSnufitBullets"]),
    ("bhvMontyMole", &["oAction", "oMontyMoleCurrentHole", "oMontyMoleHeightRelativeToFloor", "oMontyMoleHoleCooldown"]),
    ("bhvSkeeter", &["oAction", "oSkeeterTargetAngle", "oSkeeterWaitTime"]),
    ("bhvFlyGuy", &["oAction", "oFlyGuyIdleTimer", "oFlyGuyOscTimer", "oFlyGuyLungeTargetPitch"]),
    ("bhvSwoop", &["oAction", "oSwoopBonkCountdown", "oSwoopTargetPitch", "oSwoopTargetYaw"]),
    ("bhvCannon", &["oAction", "oCannonUnkF4"]),
    ("bhvYellowCoin", &["oDamageOrCoinValue", "oInteractStatus", "oPosY"]),
    ("bhvBlueCoinSliding", &["oDamageOrCoinValue", "oInteractStatus"]),
    ("bhvSpawnedStar", &["oAction", "oStarSpawnDisFromHome", "oForwardVel", "oVelY", "oInteractStatus", "oTimer"]),
    ("bhvStar", &["oAction", "oInteractStatus", "oBehParams2ndByte"]),
    ("bhvGrandStar", &["oAction", "oInteractStatus"]),
    ("bhvGiantPole", &["oPosY", "oHomeY", "hitboxHeight", "hitboxRadius"]),
    ("bhvBreakableBox", &["oAction", "oInteractStatus", "oBreakableBoxSmallReleased"]),
    ("bhvExclamationBox", &["oAction", "oExclamationBoxUnkF4", "oBehParams2ndByte"]),
    ("bhvBreakableBoxSmall", &["oAction", "oBreakableBoxSmallReleased", "oBreakableBoxSmallFramesSinceReleased", "oHeldState"]),
    ("bhvRrElevatorPlatform", PLAT_KEYS),
    ("bhvHmcElevatorPlatform", PLAT_KEYS),
    ("bhvWdwExpressElevator", PLAT_KEYS),
    ("bhvWdwExpressElevatorPlatform", PLAT_KEYS),
    ("bhvMeshElevator", PLAT_KEYS),
    ("bhvCheckerboardElevatorGroup", PLAT_KEYS),
    ("bhvCheckerboardPlatformSub", &["oAction", "oPosY", "oHomeY", "oVelY", "oForwardVel", "oCheckerBoardPlatformUnkF8", "oCheckerBoardPlatformUnkFC"]),
    ("bhvSeesawPlatform", &["oSeesawPlatformPitchVel", "oFaceAnglePitch", "oAngleVelPitch", "oAction"]),
    ("bhvDddMovingPole", &["oDDDPoleVel", "oDDDPoleMaxOffset", "oDDDPoleOffset"]),
    ("bhvSwingPlatform", &["oSwingPlatformAngle", "oSwingPlatformSpeed"]),
    ("bhvWfSlidingTowerPlatform", &["oWFSlidBrickPtfmMovVel", "oAction", "oPosX", "oPosZ"]),
    ("bhvTumblingBridgePlatform", &["oAction", "oTimer"]),
    ("bhvBitfsTiltingInvertedPyramid", &["oTiltingPyramidNormalX", "oTiltingPyramidNormalY", "oTiltingPyramidNormalZ", "oTiltingPyramidMarioOnPlatform"]),
    ("bhvBitfsSinkingPlatforms", &["oBitfsPlatformTimer", "oAction", "oPosY"]),
    ("bhvFloorSwitchAnimatesObject", &["oAction", "oFloorSwitchPressAnimationUnkF4"]),
    ("bhvBlueCoinSwitch", &["oAction", "oTimer"]),
    ("bhvDorrie", &["oDorrieDistToHome", "oDorrieOffsetY", "oDorrieVelY", "oDorrieLiftingMario", "oDorrieGroundPounded", "oAction"]),
    ("bhvWaterLevelPillar", &["oAction"]),
    ("bhvCapSwitch", &["oAction", "oInteractStatus"]),
    ("bhvDoor", &["oAction", "oDoorUnkF8", "oInteractStatus"]),
    ("bhvUkiki", &["oAction", "oHeldState"]),
    ("bhvPokey", &["oPokeyNumAliveBodyParts", "oPokeyAliveBodyPartFlags", "oPokeyHeadWasKilled", "oAction"]),
    ("bhvMips", &["oMipsStarStatus", "oMipsStartWaypointIndex", "oMipsForwardVelocity", "oAction"]),
    ("bhvRacingPenguin", &["oRacingPenguinInitTextCooldown", "oRacingPenguinMarioWon", "oRacingPenguinReachedBottom", "oRacingPenguinMarioCheated", "oAction"]),
    ("bhvTweester", &["oAction", "oTimer"]),
    ("bhvToxBox", &["oToxBoxMovementPattern", "oToxBoxMovementStep", "oAction"]),
    ("bhvEyerokBoss", &["oEyerokBossNumHands", "oEyerokBossActiveHand", "oAction", "oHealth"]),
    ("bhvEyerokHand", &["oEyerokHandWakeUpTimer", "oEyerokReceivedAttack", "oAction"]),
    ("bhvBigBully", &["oBullySubtype", "oBullyKBTimerAndMinionKOCounter", "oBullyMarioCollisionAngle", "oAction", "oForwardVel"]),
    ("bhvHiddenBlueCoin", &["oDamageOrCoinValue", "oInteractStatus", "oHiddenBlueCoinSwitch", "oAction"]),
];

/// The key fields of a behaviour (empty if none curated).
pub fn keys_of(name: &str) -> &'static [&'static str] {
    KEYS.iter().find(|(n, _)| *n == name).map_or(&[], |(_, k)| *k)
}
