// Generated from sm64-port (US) level data by a parser over
// levels/<level>/script.c (OBJECT / OBJECT_WITH_ACTS, following JUMP_LINKed
// script_func_local_* into their AREA) and levels/<level>/areas/*/macro.inc.c
// (macro_box_star_N = ! box with star index N-1, include/macro_presets.h),
// with names from text/us/courses.h. Every star whose spot is fixed in the
// level data:
// - Star: a placed bhvStar (always loaded in its area and acts).
// - Box: an ! box holding a star (bhvExclamationBox, 2nd byte 8 or 10-14,
//   src/game/behaviors/exclamation_box.inc.c; index = byte 0 | content's
//   star). JRB's stone pillar is a box in US (bhvStar only in JP/SH).
// - RedCoinStar: bhvHiddenRedCoinStar / bhvBowserCourseRedCoinStar; its star
//   appears at the marker once all 8 red coins are collected
//   (src/game/behaviors/spawn_star.inc.c, hidden_star.inc.c). Every such
//   area holds exactly 8 bhvRedCoin.
// - HiddenStar: bhvHiddenStar; its star appears once all 5
//   bhvHiddenStarTrigger (secrets) in the area are touched (hidden_star.inc.c).
// Acts: bit n-1 = act n (ACT_n masks), 0x3f = ALL_ACTS.
use obrrt_core::layout::*;

use super::{Spot, SpotKind::*};

pub const SPOTS: &[Spot] = &[
    Spot { course: 1, index: 3, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [-6000.0, 1000.0, 2400.0], kind: RedCoinStar, name: "FIND THE 8 RED COINS" },
    Spot { course: 1, index: 4, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [-6600.0, 1000.0, 1250.0], kind: HiddenStar, name: "MARIO WINGS TO THE SKY" },
    Spot { course: 1, index: 5, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [1550.0, 1200.0, 300.0], kind: Star, name: "BEHIND CHAIN CHOMP'S GATE" },
    Spot { course: 1, index: 2, level: LEVEL_BOB, area: 1, acts: 0x3f, pos: [5540.0, 3350.0, 1200.0], kind: Box, name: "SHOOT TO THE ISLAND IN THE SKY" },
    Spot { course: 2, index: 1, level: LEVEL_WF, area: 1, acts: 0x3e, pos: [300.0, 5550.0, 0.0], kind: Star, name: "TO THE TOP OF THE FORTRESS" },
    Spot { course: 2, index: 2, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [-2500.0, 1500.0, -750.0], kind: Star, name: "SHOOT INTO THE WILD BLUE" },
    Spot { course: 2, index: 3, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [4600.0, 550.0, 2500.0], kind: RedCoinStar, name: "RED COINS ON THE FLOATING ISLE" },
    Spot { course: 2, index: 4, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [2880.0, 4300.0, 190.0], kind: Star, name: "FALL ONTO THE CAGED ISLAND" },
    Spot { course: 2, index: 5, level: LEVEL_WF, area: 1, acts: 0x3f, pos: [590.0, 2450.0, 2650.0], kind: Star, name: "BLAST AWAY THE WALL" },
    Spot { course: 3, index: 3, level: LEVEL_JRB, area: 1, acts: 0x3f, pos: [4900.0, 2400.0, 800.0], kind: RedCoinStar, name: "RED COINS ON THE SHIP AFLOAT" },
    Spot { course: 3, index: 4, level: LEVEL_JRB, area: 1, acts: 0x3f, pos: [1540.0, 2160.0, 2130.0], kind: Box, name: "BLAST TO THE STONE PILLAR" },
    Spot { course: 3, index: 5, level: LEVEL_JRB, area: 1, acts: 0x3e, pos: [5000.0, -4800.0, 2500.0], kind: Star, name: "THROUGH THE JET STREAM" },
    Spot { course: 3, index: 0, level: LEVEL_JRB, area: 2, acts: 0x3f, pos: [0.0, 1600.0, 3000.0], kind: Box, name: "PLUNDER IN THE SUNKEN SHIP" },
    Spot { course: 4, index: 3, level: LEVEL_CCM, area: 1, acts: 0x3f, pos: [4200.0, -927.0, 400.0], kind: RedCoinStar, name: "FROSTY SLIDE FOR 8 RED COINS" },
    Spot { course: 4, index: 5, level: LEVEL_CCM, area: 1, acts: 0x3f, pos: [-2000.0, -2200.0, -3000.0], kind: Star, name: "WALL KICKS WILL WORK" },
    Spot { course: 5, index: 2, level: LEVEL_BBH, area: 1, acts: 0x3f, pos: [-2030.0, 1350.0, 1940.0], kind: Star, name: "SECRET OF THE HAUNTED BOOKS" },
    Spot { course: 5, index: 3, level: LEVEL_BBH, area: 1, acts: 0x3f, pos: [-204.0, 1100.0, 1576.0], kind: RedCoinStar, name: "SEEK THE 8 RED COINS" },
    Spot { course: 6, index: 0, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-3600.0, -4000.0, 3600.0], kind: Star, name: "SWIMMING BEAST IN THE CAVERN" },
    Spot { course: 6, index: 1, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [4000.0, 300.0, 5000.0], kind: RedCoinStar, name: "ELEVATE FOR 8 RED COINS" },
    Spot { course: 6, index: 2, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [6200.0, -4400.0, 2300.0], kind: Star, name: "METAL-HEAD MARIO CAN MOVE!" },
    Spot { course: 6, index: 3, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-2100.0, 2100.0, -7550.0], kind: Star, name: "NAVIGATING THE TOXIC MAZE" },
    Spot { course: 6, index: 4, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-6500.0, 2700.0, -1600.0], kind: Star, name: "A-MAZE-ING EMERGENCY EXIT" },
    Spot { course: 6, index: 5, level: LEVEL_HMC, area: 1, acts: 0x3f, pos: [-5000.0, 3050.0, -6700.0], kind: Star, name: "WATCH FOR ROLLING ROCKS" },
    Spot { course: 7, index: 2, level: LEVEL_LLL, area: 1, acts: 0x3f, pos: [-4400.0, 350.0, 250.0], kind: RedCoinStar, name: "8-COIN PUZZLE WITH 15 PIECES" },
    Spot { course: 7, index: 3, level: LEVEL_LLL, area: 1, acts: 0x3f, pos: [3100.0, 400.0, 7900.0], kind: Star, name: "RED-HOT LOG ROLLING" },
    Spot { course: 7, index: 4, level: LEVEL_LLL, area: 2, acts: 0x3f, pos: [2523.0, 3850.0, -901.0], kind: Star, name: "HOT-FOOT-IT INTO THE VOLCANO" },
    Spot { course: 7, index: 5, level: LEVEL_LLL, area: 2, acts: 0x3f, pos: [1800.0, 3400.0, 1450.0], kind: Star, name: "ELEVATOR TOUR IN THE VOLCANO" },
    Spot { course: 8, index: 1, level: LEVEL_SSL, area: 1, acts: 0x3f, pos: [-2050.0, 1200.0, -580.0], kind: Star, name: "SHINING ATOP THE PYRAMID" },
    Spot { course: 8, index: 4, level: LEVEL_SSL, area: 1, acts: 0x3f, pos: [6000.0, 800.0, 3500.0], kind: RedCoinStar, name: "FREE FLYING FOR 8 RED COINS" },
    Spot { course: 8, index: 2, level: LEVEL_SSL, area: 2, acts: 0x3f, pos: [500.0, 5050.0, -500.0], kind: Star, name: "INSIDE THE ANCIENT PYRAMID" },
    Spot { course: 8, index: 5, level: LEVEL_SSL, area: 2, acts: 0x3f, pos: [900.0, 1400.0, 2350.0], kind: HiddenStar, name: "PYRAMID PUZZLE" },
    Spot { course: 9, index: 0, level: LEVEL_DDD, area: 2, acts: 0x3f, pos: [3900.0, 850.0, -600.0], kind: Star, name: "BOARD BOWSER'S SUB" },
    Spot { course: 9, index: 2, level: LEVEL_DDD, area: 2, acts: 0x3f, pos: [5513.0, 1200.0, 900.0], kind: RedCoinStar, name: "POLE-JUMPING FOR RED COINS" },
    Spot { course: 9, index: 5, level: LEVEL_DDD, area: 2, acts: 0x3f, pos: [2030.0, -3700.0, -2780.0], kind: Star, name: "COLLECT THE CAPS..." },
    Spot { course: 10, index: 0, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [700.0, 4500.0, 690.0], kind: Star, name: "SNOWMAN'S BIG HEAD" },
    Spot { course: 10, index: 2, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [4350.0, 1350.0, 4350.0], kind: Star, name: "IN THE DEEP FREEZE" },
    Spot { course: 10, index: 4, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [5000.0, 1200.0, 0.0], kind: RedCoinStar, name: "SHELL SHREDDIN' FOR RED COINS" },
    Spot { course: 10, index: 3, level: LEVEL_SL, area: 1, acts: 0x3f, pos: [-4700.0, 1300.0, 5850.0], kind: Box, name: "WHIRL FROM THE FREEZING POND" },
    Spot { course: 10, index: 5, level: LEVEL_SL, area: 2, acts: 0x3f, pos: [0.0, 500.0, 1000.0], kind: Star, name: "INTO THE IGLOO" },
    Spot { course: 11, index: 2, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [3360.0, 1580.0, 2660.0], kind: HiddenStar, name: "SECRETS IN THE SHALLOWS & SKY" },
    Spot { course: 11, index: 3, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [890.0, 3400.0, -2040.0], kind: Star, name: "EXPRESS ELEVATOR--HURRY UP!" },
    Spot { course: 11, index: 0, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [-2200.0, 2600.0, 3500.0], kind: Box, name: "SHOCKING ARROW LIFTS!" },
    Spot { course: 11, index: 1, level: LEVEL_WDW, area: 1, acts: 0x3f, pos: [1550.0, 4350.0, 100.0], kind: Box, name: "TOP O' THE TOWN" },
    Spot { course: 11, index: 4, level: LEVEL_WDW, area: 2, acts: 0x3f, pos: [-770.0, -1600.0, 3600.0], kind: RedCoinStar, name: "GO TO TOWN FOR RED COINS" },
    Spot { course: 11, index: 5, level: LEVEL_WDW, area: 2, acts: 0x3f, pos: [2180.0, -840.0, 3720.0], kind: Star, name: "QUICK RACE THROUGH DOWNTOWN!" },
    Spot { course: 12, index: 0, level: LEVEL_TTM, area: 1, acts: 0x1, pos: [1200.0, 2600.0, 150.0], kind: Star, name: "SCALE THE MOUNTAIN" },
    Spot { course: 12, index: 2, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [-3250.0, -2500.0, -3700.0], kind: RedCoinStar, name: "SCARY 'SHROOMS, RED COINS" },
    Spot { course: 12, index: 3, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [-2900.0, -2700.0, 3650.0], kind: Star, name: "MYSTERIOUS MOUNTAINSIDE" },
    Spot { course: 12, index: 4, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [1800.0, 1200.0, 1050.0], kind: Star, name: "BREATHTAKING VIEW FROM BRIDGE" },
    Spot { course: 12, index: 5, level: LEVEL_TTM, area: 1, acts: 0x3f, pos: [7300.0, -3100.0, 1300.0], kind: Star, name: "BLAST TO THE LONELY MUSHROOM" },
    Spot { course: 13, index: 1, level: LEVEL_THI, area: 1, acts: 0x3f, pos: [2600.0, 3500.0, -2400.0], kind: Box, name: "THE TIP TOP OF THE HUGE ISLAND" },
    Spot { course: 13, index: 3, level: LEVEL_THI, area: 2, acts: 0x3f, pos: [0.0, -700.0, -4500.0], kind: HiddenStar, name: "FIVE ITTY BITTY SECRETS" },
    Spot { course: 13, index: 4, level: LEVEL_THI, area: 3, acts: 0x3f, pos: [-1800.0, 800.0, -1500.0], kind: RedCoinStar, name: "WIGGLER'S RED COINS" },
    Spot { course: 14, index: 0, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1450.0, -1130.0, -1050.0], kind: Star, name: "ROLL INTO THE CAGE" },
    Spot { course: 14, index: 1, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1850.0, 300.0, -950.0], kind: Star, name: "THE PIT AND THE PENDULUMS" },
    Spot { course: 14, index: 2, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1300.0, -2250.0, -1300.0], kind: Star, name: "GET A HAND" },
    Spot { course: 14, index: 3, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [2200.0, 7300.0, 2210.0], kind: Star, name: "STOMP ON THE THWOMP" },
    Spot { course: 14, index: 4, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [-1050.0, 2400.0, -790.0], kind: Star, name: "TIMED JUMPS ON MOVING BARS" },
    Spot { course: 14, index: 5, level: LEVEL_TTC, area: 1, acts: 0x3f, pos: [1815.0, -3200.0, 800.0], kind: RedCoinStar, name: "STOP TIME FOR RED COINS" },
    Spot { course: 15, index: 0, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [1450.0, 3400.0, -2352.0], kind: Star, name: "CRUISER CROSSING THE RAINBOW" },
    Spot { course: 15, index: 1, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [-4200.0, 6700.0, -4450.0], kind: Star, name: "THE BIG HOUSE IN THE SKY" },
    Spot { course: 15, index: 2, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [-5150.0, -1400.0, 0.0], kind: RedCoinStar, name: "COINS AMASSED IN A MAZE" },
    Spot { course: 15, index: 3, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [-5850.0, -700.0, 4950.0], kind: Star, name: "SWINGIN' IN THE BREEZE" },
    Spot { course: 15, index: 4, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [3700.0, -400.0, 6600.0], kind: Star, name: "TRICKY TRIANGLES!" },
    Spot { course: 15, index: 5, level: LEVEL_RR, area: 1, acts: 0x3f, pos: [5000.0, 4100.0, 4440.0], kind: Box, name: "SOMEWHERE OVER THE RAINBOW" },
    Spot { course: 16, index: 0, level: LEVEL_BITDW, area: 1, acts: 0x3f, pos: [7180.0, 3000.0, 0.0], kind: RedCoinStar, name: "BOWSER IN THE DARK WORLD" },
    Spot { course: 17, index: 0, level: LEVEL_BITFS, area: 1, acts: 0x3f, pos: [1200.0, 5700.0, 160.0], kind: RedCoinStar, name: "BOWSER IN THE FIRE SEA" },
    Spot { course: 18, index: 0, level: LEVEL_BITS, area: 1, acts: 0x3f, pos: [350.0, 6800.0, -6800.0], kind: RedCoinStar, name: "BOWSER IN THE SKY" },
    Spot { course: 19, index: 0, level: LEVEL_PSS, area: 1, acts: 0x3f, pos: [-6385.0, -4200.0, 5770.0], kind: Box, name: "THE PRINCESS'S SECRET SLIDE" },
    Spot { course: 20, index: 0, level: LEVEL_COTMC, area: 1, acts: 0x3f, pos: [0.0, -200.0, -7000.0], kind: RedCoinStar, name: "CAVERN OF THE METAL CAP" },
    Spot { course: 21, index: 0, level: LEVEL_TOTWC, area: 1, acts: 0x3f, pos: [800.0, -1700.0, 0.0], kind: RedCoinStar, name: "TOWER OF THE WING CAP" },
    Spot { course: 22, index: 0, level: LEVEL_VCUTM, area: 1, acts: 0x3f, pos: [4600.0, 250.0, -4500.0], kind: RedCoinStar, name: "VANISH CAP UNDER THE MOAT" },
    Spot { course: 23, index: 0, level: LEVEL_WMOTR, area: 1, acts: 0x3f, pos: [-160.0, 1950.0, -470.0], kind: RedCoinStar, name: "WING MARIO OVER THE RAINBOW" },
    Spot { course: 24, index: 0, level: LEVEL_SA, area: 1, acts: 0x3f, pos: [0.0, -4250.0, 0.0], kind: RedCoinStar, name: "THE SECRET AQUARIUM" },
];
