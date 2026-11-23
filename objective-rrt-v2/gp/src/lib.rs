//! Genetic programming in Koza's sense for Mario movement controllers:
//! programs are typed if-trees over game state ([`lang`]), evolved with
//! ramped half-and-half, tournaments and subtree crossover ([`evolve`]) and
//! scored over several start states from real routes ([`rollout`],
//! [`problems`]), so a program has to generalise.

pub mod evolve;
pub mod lang;
pub mod problems;
pub mod rollout;
