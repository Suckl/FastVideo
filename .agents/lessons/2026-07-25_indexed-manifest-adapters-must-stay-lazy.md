---
date: 2026-07-25
experiment: VidaForge-3M indexed-TAR preprocessing review
category: data
severity: critical
---

# Indexed Manifest Adapters Must Stay Lazy

## What Happened

An adapter for a multi-terabyte indexed-TAR dataset normalized rows with
`Dataset.map()`. Normalization extracted, hashed, and probed every selected
video before the DataLoader yielded its first batch. A one-shard smoke test
worked, but a full run would eagerly scan and materialize the entire dataset.

## Root Cause

The implementation treated media normalization like inexpensive Arrow
metadata normalization. Indexed media access is an I/O boundary: reading one
row can copy and decode megabytes, so its placement determines startup time,
disk consumption, recovery behavior, and worker parallelism.

## Fix / Workaround

Keep metadata filtering and distributed sharding eager, but return a
`torch.utils.data.IterableDataset` that reads, verifies, and probes media only
when a DataLoader worker consumes that row. Partition again by DataLoader
worker. For decoders that accept encoded bytes, pass verified bytes directly
and make persistent materialization opt-in.

## Prevention

- Classify every manifest transform as metadata-only or media-I/O.
- Never put media extraction or decoding in an eager whole-dataset `map()`.
- Test that constructing the dataset performs no video materialization.
- Test worker partitioning so iterable datasets do not duplicate samples.
- Estimate both source and derived storage before claiming full-dataset
  support.
