HARDWARE EVENTS - device identity across card removal and PCIe repair
=====================================================================

This file exists because the box's GPU population changed underneath a
benchmark campaign and nothing in the record said so. Two entries follow: a
physical card removal, and a PCIe link-speed repair. Both silently changed the
meaning of numbers this branch quotes.

WHY IT MATTERS
--------------
Benchmark ledgers on this branch identify cards by ROCm device INDEX
("ROCm0,ROCm1"). An index is a position in the current enumeration, not a
physical card. When a card is removed, the remaining cards are renumbered and
every bare index silently starts pointing at different hardware. Two ledgers
months apart can say "ROCm0,ROCm1" and mean four different pieces of silicon.

Consequence in force from this entry forward: ANY comparison that spans one of
the events below and leans on a device index is ambiguous. Absolute throughput
figures are build- AND fabric-specific; compare within a single era only.


EVENT 1 - PHYSICAL GPU REMOVAL
------------------------------
Observed by: the fourth device index stopped appearing in the test system.

Before (4-card census, 2026-11-13):
  index 0  PCI 12:00.0
  index 1  PCI 17:00.0
  index 2  PCI 1a:00.0
  index 3  PCI 1d:00.0

After (3 cards, current):
  index 0  PCI 15:00.0
  index 1  PCI 18:00.0
  index 2  PCI 1b:00.0

Both 12:00.0 and 1d:00.0 are absent. The remaining cards did not keep their
indices; they re-enumerated into new slots. So index 0 today is NOT index 0 of
the 4-card era, and index 1 today is not that era's index 1 either.

One of the removed cards was the ECC-degraded unit that had been quarantined
from measurement pairs. Its removal therefore also changed the POPULATION, not
just the numbering.

Durable identity. ROCm.ioctl unique IDs, current index order 0,1,2:
  0x214c20c173497dd3
  0x20a70a17348c2f7
  0x22228617348c2f7
The pre-removal mapping from index to unique ID was NOT recorded and cannot be
reconstructed from anything on the box. That gap is permanent; do not paper
over it by assuming order was preserved.

Annotate when reading old ledgers:
  - legs dated before the removal may include the ECC-degraded card;
  - legs dated after it do not, and ran on whichever two or three survivors
    the enumeration happened to expose.


EVENT 2 - PCIe LINK SPEED REPAIR
--------------------------------
The fabric was running at Gen3 with an x8 width cap and was later repaired to
Gen4. Width did not change.

  achieved before : ~7.9 GB/s per direction (Gen3 x8 ceiling)
  achieved after  : 13.38-13.39 GB/s host-to-device, 13.28 GB/s device-to-host
  theoretical     : Gen4 x8 = 15.75 GB/s, Gen3 x8 = 7.88 GB/s, Gen4 x16 = 31.5 GB/s

Measured on all three cards individually (512-768 MB transfers, HIP events,
TheRock runtime). All three are IDENTICAL to within 0.01 GB/s, which is itself
a result: it rules out a per-card bandwidth difference as the cause of any
per-card performance asymmetry.

IMPORTANT - the endpoint lies. Each card's own link reports 16.0 GT/s and width
x16, which looks like a healthy Gen4 x16 slot. The bottleneck is the UPSTREAM
SWITCH PORT: every card sits behind a port whose negotiated width is x8 and
whose maximum width is also x8, so the cap is a hard capability limit, not a
failed negotiation that a retrain could lift. Read the whole parent chain, not
the endpoint. Achieved bandwidth lands at ~85 pct of the Gen4 x8 theoretical
figure, consistent with that ceiling.

Consequence: every throughput number taken before the repair was measured on a
fabric roughly half as fast. That is a HARDWARE variable that no build-level
comparison controls for. A pre-repair absolute and a post-repair absolute are
not comparable without accounting for it.


STANDING RULES
--------------
1. Every benchmark leg header must witness PHYSICAL identity (the unique ID),
   not a bare index.
2. Record a dated event entry here whenever the GPU population, the slot
   assignment, or the link state changes.
3. Never quote a cross-era absolute as a regression or a win without naming
   the era and checking both events above.
4. Endpoint link state is not fabric bandwidth. Walk the parent chain to the
   root complex before asserting any PCIe capability.

WHAT IS NOT KNOWN
-----------------
- The exact date of the removal. Only "when the fourth index disappeared" is
  established; no dated log of the physical event survived.
- The pre-removal index-to-unique-ID mapping.
- Whether any specific historical leg ran on the ECC-degraded card, except
  where that leg's own header says so.
- The cause of the link-speed regression before the repair.
