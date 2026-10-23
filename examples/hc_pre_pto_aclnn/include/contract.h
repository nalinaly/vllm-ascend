// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#ifndef HCPRE_DEVICE
  #define HCPRE_DEVICE
#endif
namespace hcpre {
struct Config {
  int64_t tokens, coreGroups, key, tile, parts, rms, fifoElements;
  int64_t padded, units, sink, rows, extraRows, pairs, workers;
  int64_t workersPerPair, extraPairs, prefix, pairsPerWorker, extraWorkers;
  uint64_t partsBytes, rmsBytes, workspaceBytes;
};
HCPRE_DEVICE inline Config MakeConfig(int64_t t, int64_t coreGroups = 24) {
  Config c{};
  c.tokens = t;
  c.coreGroups = coreGroups;
  c.key = t <= 32 ? 0 : t <= 48 ? 1 : t < 96 ? 2 : t == 96 ? 3 : t <= 144 ? 4 : 5;
  c.tile = c.key == 1 || c.key == 4 ? 48 : c.key == 5 ? 64 : 32;
  c.parts = c.key == 0 ? 11 : c.key == 1 ? 16 : 8;
  c.rms = c.key == 3 ? 16 : c.parts;
  c.fifoElements = c.key == 0 ? 32768 : (c.key == 1 || c.key == 4) ? 49152 : 65536;
  c.padded = (t + c.tile - 1) / c.tile * c.tile;
  c.units = c.padded / c.tile * c.parts;
  c.sink = t < coreGroups ? t : coreGroups;
  c.rows = t / c.sink;
  c.extraRows = t % c.sink;
  c.pairs = (t + 1) / 2;
  c.workers = 2 * coreGroups - c.sink < c.pairs * 8 ? 2 * coreGroups - c.sink : c.pairs * 8;
  c.workersPerPair = c.workers / c.pairs > 1 ? c.workers / c.pairs : 1;
  c.extraPairs = c.workers % c.pairs;
  c.prefix = c.extraPairs * (c.workersPerPair + 1);
  c.pairsPerWorker = c.pairs / c.workers;
  c.extraWorkers = c.pairs % c.workers;
  c.partsBytes = (c.parts * c.padded * 32 * 4 + 511) / 512 * 512;
  c.rmsBytes = (c.rms * c.padded * 4 + 511) / 512 * 512;
  c.workspaceBytes = c.partsBytes + c.rmsBytes + coreGroups * c.fifoElements * 4;
  return c;
}
}  // namespace hcpre
