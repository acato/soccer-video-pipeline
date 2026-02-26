# ADR-005: Reel Plugin Architecture

**Status:** Proposed
**Date:** 2026-02-26
**Author:** Architect Agent

## Problem Statement

The current pipeline couples three distinct concerns into a single rigid flow:

1. **Detection** — What happened? (YOLOv8 + heuristics produce Events)
2. **Reel selection** — Which events belong on which reel? (scattered across EVENT_REEL_MAP, `reel_targets` at creation time, and `reel_label_for()`)
3. **Clip assembly** — How to cut and pad? (hardcoded per-reel-type overrides in `_run_pipeline()`)

Adding a new reel variant (e.g., "GK distributions only", "opponent shots", "set pieces") requires touching 4-5 files: `models.py`, `event_classifier.py`, `worker.py`, `clipper.py`, and tests. This makes it impractical to rapidly prototype different filters.

### Specific Coupling Points

| Concern | Current location | Problem |
|---------|-----------------|---------|
| Which events go on keeper reel | `Event.reel_targets` set at creation + `EVENT_REEL_MAP` | Two different assignment mechanisms |
| Keeper padding (3.0/2.0) | `_run_pipeline()` lines 192-194 | Hardcoded in orchestrator, not with filter |
| Max clip duration (15s keeper / 90s highlights) | `_run_pipeline()` line 196 | Same — belongs with filter |
| GK-only confidence thresholds | `EVENT_CONFIDENCE_THRESHOLDS` in models.py | Global dict, can't vary per reel |
| Reel type iteration | `for reel_type in job.reel_types` in worker.py | Fixed set, no extension point |

## Decision

Introduce a **ReelPlugin** abstraction: a self-contained unit that declares a reel, selects events for it, and specifies how clips should be cut. Plugins are pure event filters — they do not run detection, they consume its output.

## Architecture

### Core Abstractions

```
                  ┌──────────────────────┐
                  │   Detection Stage    │  (unchanged)
                  │  PlayerDetector      │
                  │  GoalkeeperDetector  │
                  │  HighlightsClassifier│
                  └──────────┬───────────┘
                             │
                       list[Event]
                             │
                  ┌──────────▼───────────┐
                  │    EventLog (JSONL)   │  (unchanged)
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │   Plugin Registry     │  ← NEW
                  │                       │
                  │  ┌─────────────────┐  │
                  │  │ KeeperSaves     │──┼──→ keeper reel
                  │  ├─────────────────┤  │
                  │  │ KeeperDistrib   │──┼──→ keeper reel  (merged)
                  │  ├─────────────────┤  │
                  │  │ HighlightsShots │──┼──→ highlights reel
                  │  ├─────────────────┤  │
                  │  │ SetPieces       │──┼──→ highlights reel  (merged)
                  │  ├─────────────────┤  │
                  │  │ OpponentShots   │──┼──→ opponent_shots reel  (new)
                  │  └─────────────────┘  │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Clipper + Composer   │  (unchanged internals)
                  └──────────────────────┘
```

### Plugin Interface

```python
# src/reel_plugins/base.py

@dataclass(frozen=True)
class ClipParams:
    """How this plugin wants its clips cut."""
    pre_pad_sec: float = 3.0
    post_pad_sec: float = 3.0
    merge_gap_sec: float = 2.0
    max_clip_duration_sec: float = 60.0
    max_reel_duration_sec: float = 1200.0   # 20 min
    min_clip_duration_sec: float = 2.0

@dataclass(frozen=True)
class PipelineContext:
    """Read-only metadata available to plugins during selection."""
    video_duration_sec: float
    match_config: MatchConfig | None
    keeper_track_ids: dict[str, int | None]   # {"keeper_a": 42, "keeper_b": 7}
    job_id: str

class ReelPlugin(ABC):
    """A plugin that contributes clips to a named reel."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name, e.g. 'keeper_saves'."""

    @property
    @abstractmethod
    def reel_name(self) -> str:
        """Target reel identifier. Multiple plugins can share a reel_name;
        their clips are merged into a single reel."""

    @property
    def clip_params(self) -> ClipParams:
        """Override to customize padding, merge gap, duration caps."""
        return ClipParams()

    @abstractmethod
    def select_events(self, events: list[Event], ctx: PipelineContext) -> list[Event]:
        """Filter the full event log to events this plugin wants.

        This is a PURE FUNCTION: no side effects, no I/O.
        Return a subset of the input events (or empty list to contribute nothing).
        """

    def post_filter_clips(self, clips: list[ClipBoundary]) -> list[ClipBoundary]:
        """Optional hook to reorder, drop, or annotate clips after clipping.
        Default: return as-is."""
        return clips
```

### Plugin Registry

```python
# src/reel_plugins/registry.py

class PluginRegistry:
    """Discovers and manages reel plugins."""

    _plugins: dict[str, ReelPlugin]   # keyed by plugin.name

    def register(self, plugin: ReelPlugin) -> None: ...
    def get_plugins_for_reel(self, reel_name: str) -> list[ReelPlugin]: ...
    def get_all_reel_names(self) -> set[str]: ...

    @classmethod
    def from_config(cls, plugin_names: list[str]) -> "PluginRegistry":
        """Build registry from list of plugin names (e.g. from job config or env var).
        Looks up plugins in BUILTIN_PLUGINS dict."""

    @classmethod
    def default(cls) -> "PluginRegistry":
        """Registry with all built-in plugins, matching current behavior."""
```

### Built-in Plugins (migrated from current code)

```python
# src/reel_plugins/keeper.py

class KeeperSavesPlugin(ReelPlugin):
    """GK save events → keeper reel. Replaces current keeper reel logic."""
    name = "keeper_saves"
    reel_name = "keeper"
    clip_params = ClipParams(pre_pad_sec=3.0, post_pad_sec=2.0,
                             max_clip_duration_sec=15.0)

    def select_events(self, events, ctx):
        SAVE_TYPES = {EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
                      EventType.PUNCH, EventType.CATCH}
        return [e for e in events
                if e.event_type in SAVE_TYPES
                and e.is_goalkeeper_event
                and _targets_keeper_reel(e.reel_targets)]


class KeeperDistributionPlugin(ReelPlugin):
    """GK distribution events → keeper reel."""
    name = "keeper_distribution"
    reel_name = "keeper"
    clip_params = ClipParams(pre_pad_sec=1.0, post_pad_sec=3.0,
                             max_clip_duration_sec=10.0)

    def select_events(self, events, ctx):
        DIST_TYPES = {EventType.GOAL_KICK, EventType.DISTRIBUTION_SHORT,
                      EventType.DISTRIBUTION_LONG}
        return [e for e in events
                if e.event_type in DIST_TYPES
                and e.is_goalkeeper_event
                and _targets_keeper_reel(e.reel_targets)]


# src/reel_plugins/highlights.py

class HighlightsShotsPlugin(ReelPlugin):
    """Shot/goal events → highlights reel. Replaces EVENT_REEL_MAP filtering."""
    name = "highlights_shots"
    reel_name = "highlights"
    clip_params = ClipParams(pre_pad_sec=3.0, post_pad_sec=5.0,
                             max_clip_duration_sec=90.0)

    def select_events(self, events, ctx):
        SHOT_TYPES = {EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
                      EventType.GOAL, EventType.NEAR_MISS, EventType.PENALTY,
                      EventType.FREE_KICK_SHOT}
        return [e for e in events
                if e.event_type in SHOT_TYPES
                and "highlights" in e.reel_targets]
```

### Example Prototype Plugin (new capability)

```python
# src/reel_plugins/custom/opponent_shots.py

class OpponentShotsPlugin(ReelPlugin):
    """Shots faced by our keeper — useful for coach review."""
    name = "opponent_shots"
    reel_name = "opponent_shots"           # new reel!
    clip_params = ClipParams(pre_pad_sec=5.0, post_pad_sec=3.0,
                             max_clip_duration_sec=30.0)

    def select_events(self, events, ctx):
        return [e for e in events
                if e.event_type in {EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET}
                and not e.is_goalkeeper_event]     # shot event, not GK event
```

This took <5 minutes to write. No changes to detection, models, or worker.

### Reel Assembly Loop (refactored _run_pipeline)

```python
# In _run_pipeline(), SEGMENTING stage — replaces current hardcoded logic:

registry = PluginRegistry.from_config(job.plugin_names or cfg.DEFAULT_PLUGINS)
all_events = event_log.read_all()
ctx = PipelineContext(video_duration_sec=duration, match_config=match_cfg,
                      keeper_track_ids=runner.keeper_ids, job_id=job.job_id)

clips_by_reel: dict[str, list[ClipBoundary]] = {}

for reel_name in registry.get_all_reel_names():
    plugins = registry.get_plugins_for_reel(reel_name)
    reel_clips = []
    for plugin in plugins:
        selected = plugin.select_events(all_events, ctx)
        if not selected:
            continue
        p = plugin.clip_params
        clips = compute_clips(selected, duration, reel_name,
                              p.pre_pad_sec, p.post_pad_sec,
                              p.merge_gap_sec, p.max_clip_duration_sec)
        clips = plugin.post_filter_clips(clips)
        reel_clips.extend(clips)

    # Merge clips from multiple plugins, dedup overlaps
    reel_clips.sort(key=lambda c: c.start_sec)
    reel_clips = merge_overlapping_clips(reel_clips)
    reel_clips = postprocess_clips(reel_clips, max_reel_dur=max_reel_dur)
    clips_by_reel[reel_name] = reel_clips
```

## Migration Plan

### Phase 1: Foundation (no behavior change)

**Files created:**
- `src/reel_plugins/__init__.py`
- `src/reel_plugins/base.py` — `ReelPlugin` ABC, `ClipParams`, `PipelineContext`
- `src/reel_plugins/registry.py` — `PluginRegistry`

**Tests:**
- `tests/unit/test_plugin_base.py` — ABC contract, ClipParams defaults, PipelineContext immutability

### Phase 2: Migrate existing logic to plugins (behavior-preserving)

**Files created:**
- `src/reel_plugins/keeper.py` — `KeeperSavesPlugin`, `KeeperDistributionPlugin`
- `src/reel_plugins/highlights.py` — `HighlightsShotsPlugin`

**Tests:**
- `tests/unit/test_keeper_plugin.py` — event selection with synthetic events
- `tests/unit/test_highlights_plugin.py` — event selection with synthetic events
- `tests/unit/test_plugin_registry.py` — registration, lookup, default registry

**Validation:** Run existing `test_gk_pipeline.py` — must still pass unchanged. The plugins should produce identical event selections as the current hardcoded paths.

### Phase 3: Wire plugins into worker

**Files modified:**
- `src/api/worker.py` — replace hardcoded reel loop with plugin registry loop
- `src/segmentation/clipper.py` — remove reel_type-specific branching if any remains

**Files modified (cleanup):**
- `src/detection/models.py` — `EVENT_REEL_MAP` becomes documentation-only (plugins own selection)
- Worker no longer has per-reel-type `if` blocks for padding

**Tests:**
- `tests/unit/test_worker.py` — update mocks to verify plugin registry is used
- Regression: all existing tests pass

### Phase 4: Plugin discovery & job-level configuration

**Files modified:**
- `src/ingestion/models.py` — add `plugin_names: list[str] | None` to Job model
- `src/api/routes/jobs.py` — accept `plugins` in POST /jobs
- `src/config.py` — add `DEFAULT_PLUGINS` env var (comma-separated list)
- UI: plugin checkboxes or multi-select

**Tests:**
- `tests/unit/test_plugin_registry.py` — from_config with subset of plugins
- `tests/unit/test_worker.py` — job with custom plugin list

### Phase 5: Prototype plugins & examples

**Files created:**
- `src/reel_plugins/custom/` — directory for experimental plugins
- `src/reel_plugins/custom/opponent_shots.py`
- `src/reel_plugins/custom/set_pieces.py`
- `src/reel_plugins/custom/gk_one_on_ones.py`

**Tests:**
- One test file per custom plugin, pure event filtering

## Testing Strategy

### The Core Insight: Plugins Are Pure Functions

A plugin's `select_events()` takes `list[Event]` and returns `list[Event]`. No video. No FFmpeg. No ML models. No tracks. No I/O. This makes them testable in **milliseconds**.

### Test Pyramid for Plugins

```
                    ┌──────────────┐
                    │  E2E (1-2)   │  Full video → verify reel has expected clips
                    ├──────────────┤
                    │  Integration │  Events JSONL → plugin → clipper → verify boundaries
                    │    (5-10)    │
                    ├──────────────┤
                    │  Unit Tests  │  Synthetic events → plugin.select_events() → assertions
                    │   (50-100)   │  < 1ms each, no deps beyond pytest
                    └──────────────┘
```

### Unit Test Pattern

```python
# tests/unit/test_keeper_plugin.py

@pytest.fixture
def mixed_events():
    """Realistic mix of GK and non-GK events."""
    return [
        make_event(EventType.SHOT_STOP_DIVING, reel_targets=["keeper"],
                   confidence=0.85, is_gk=True, start=10.0, end=11.0),
        make_event(EventType.DISTRIBUTION_SHORT, reel_targets=["keeper"],
                   confidence=0.80, is_gk=True, start=45.0, end=46.0),
        make_event(EventType.SHOT_ON_TARGET, reel_targets=["highlights"],
                   confidence=0.90, is_gk=False, start=60.0, end=61.0),
        make_event(EventType.GOAL, reel_targets=["highlights"],
                   confidence=0.95, is_gk=False, start=75.0, end=76.0),
    ]

class TestKeeperSavesPlugin:
    def test_selects_only_save_events(self, mixed_events, ctx):
        plugin = KeeperSavesPlugin()
        selected = plugin.select_events(mixed_events, ctx)
        assert len(selected) == 1
        assert selected[0].event_type == EventType.SHOT_STOP_DIVING

    def test_ignores_distribution_events(self, mixed_events, ctx):
        plugin = KeeperSavesPlugin()
        selected = plugin.select_events(mixed_events, ctx)
        assert not any(e.event_type == EventType.DISTRIBUTION_SHORT for e in selected)

    def test_clip_params_tight_padding(self):
        plugin = KeeperSavesPlugin()
        assert plugin.clip_params.pre_pad_sec == 3.0
        assert plugin.clip_params.post_pad_sec == 2.0
        assert plugin.clip_params.max_clip_duration_sec == 15.0

    def test_empty_events_returns_empty(self, ctx):
        assert KeeperSavesPlugin().select_events([], ctx) == []

    def test_no_gk_events_returns_empty(self, ctx):
        """Non-GK events should never be selected."""
        events = [make_event(EventType.SHOT_ON_TARGET, reel_targets=["highlights"])]
        assert KeeperSavesPlugin().select_events(events, ctx) == []
```

### Integration Test: Plugin → Clipper Round-Trip

```python
# tests/unit/test_plugin_clipper_integration.py

def test_keeper_saves_plugin_produces_valid_clips():
    """Plugin event selection feeds directly into compute_clips."""
    events = [
        make_event(SHOT_STOP_DIVING, start=30.0, end=31.0, reel_targets=["keeper"]),
        make_event(SHOT_STOP_STANDING, start=85.0, end=86.0, reel_targets=["keeper"]),
    ]
    plugin = KeeperSavesPlugin()
    ctx = make_context(video_duration_sec=5400.0)
    selected = plugin.select_events(events, ctx)
    p = plugin.clip_params

    clips = compute_clips(selected, ctx.video_duration_sec, plugin.reel_name,
                          p.pre_pad_sec, p.post_pad_sec,
                          p.merge_gap_sec, p.max_clip_duration_sec)

    assert len(clips) == 2
    assert clips[0].start_sec == pytest.approx(27.0)   # 30.0 - 3.0
    assert clips[0].end_sec == pytest.approx(33.0)      # 31.0 + 2.0
    assert clips[1].start_sec == pytest.approx(82.0)    # 85.0 - 3.0


def test_multiple_plugins_same_reel_merge():
    """Two plugins contributing to 'keeper' reel produce merged, non-overlapping clips."""
    saves_events = [make_event(SHOT_STOP_DIVING, start=30.0, end=31.0, reel_targets=["keeper"])]
    dist_events = [make_event(DISTRIBUTION_SHORT, start=32.0, end=33.0, reel_targets=["keeper"])]
    all_events = saves_events + dist_events

    saves_plugin = KeeperSavesPlugin()
    dist_plugin = KeeperDistributionPlugin()
    ctx = make_context(video_duration_sec=5400.0)

    all_clips = []
    for plugin in [saves_plugin, dist_plugin]:
        selected = plugin.select_events(all_events, ctx)
        p = plugin.clip_params
        clips = compute_clips(selected, ctx.video_duration_sec, plugin.reel_name,
                              p.pre_pad_sec, p.post_pad_sec,
                              p.merge_gap_sec, p.max_clip_duration_sec)
        all_clips.extend(clips)

    all_clips.sort(key=lambda c: c.start_sec)
    merged = merge_overlapping_clips(all_clips)

    # Two events 2s apart with padding will merge into one clip
    assert len(merged) == 1
    assert merged[0].start_sec == pytest.approx(27.0)   # earliest start
    assert merged[0].end_sec == pytest.approx(36.0)      # latest end
```

### Regression Test: Behavioral Equivalence

```python
# tests/unit/test_plugin_regression.py

def test_default_plugins_match_legacy_behavior():
    """Default plugin registry produces identical clips to pre-refactor code.

    This test uses the same event fixtures as test_gk_pipeline.py and verifies
    that the plugin-based path yields the same ClipBoundary list.
    """
    events = load_fixture_events("sample_events.jsonl")

    # Legacy path (pre-refactor)
    legacy_keeper_clips = compute_clips(
        [e for e in events if any(t.startswith("keeper") for t in e.reel_targets)],
        duration=5400.0, reel_type="keeper",
        pre_pad=3.0, post_pad=2.0, merge_gap_sec=2.0, max_clip_duration_sec=15.0
    )

    # Plugin path
    registry = PluginRegistry.default()
    ctx = make_context(video_duration_sec=5400.0)
    plugin_clips = assemble_reel_clips(registry, "keeper", events, ctx)

    assert len(plugin_clips) == len(legacy_keeper_clips)
    for pc, lc in zip(plugin_clips, legacy_keeper_clips):
        assert pc.start_sec == pytest.approx(lc.start_sec)
        assert pc.end_sec == pytest.approx(lc.end_sec)
```

### Shared Test Fixtures

```python
# tests/conftest.py  (additions)

@pytest.fixture
def make_plugin_event():
    """Factory for events with minimal boilerplate."""
    def _make(event_type, start, end=None, reel_targets=None,
              confidence=0.85, is_gk=False):
        end = end or start + 1.0
        reel_targets = reel_targets or []
        return Event(
            event_id=f"evt-{uuid4().hex[:8]}",
            job_id="test-job",
            source_file="match.mp4",
            event_type=event_type,
            timestamp_start=start,
            timestamp_end=end,
            confidence=confidence,
            reel_targets=reel_targets,
            is_goalkeeper_event=is_gk,
            frame_start=int(start * 30),
            frame_end=int(end * 30),
        )
    return _make


@pytest.fixture
def make_context():
    """Factory for PipelineContext."""
    def _make(video_duration_sec=5400.0, keeper_ids=None):
        return PipelineContext(
            video_duration_sec=video_duration_sec,
            match_config=make_match_config(),
            keeper_track_ids=keeper_ids or {"keeper_a": 1, "keeper_b": None},
            job_id="test-job",
        )
    return _make
```

### What This Enables

| Scenario | Before (current) | After (plugins) |
|----------|-------------------|-----------------|
| "Show me only GK distributions" | Edit worker.py + clipper.py + reprocess | Write 10-line plugin, select in UI |
| "Different padding for saves vs distributions" | Impossible (one padding per reel_type) | Each plugin has its own ClipParams |
| "Add opponent shots reel" | Add EVENT_REEL_MAP entry, new reel_type in worker, new padding block | Write 15-line plugin |
| "Test a new filter idea" | Process full video, wait 30+ min | Write plugin + unit test, run in <1s |
| "A/B test two highlight strategies" | Copy-paste + modify worker.py | Two plugins, swap via config |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Multiple plugins select same event → duplicate clips | `merge_overlapping_clips()` already handles this; runs after all plugins contribute |
| Plugin with bad ClipParams produces empty/bloated reel | Validation in ClipParams (`__post_init__` checks ranges); max_reel_duration cap |
| Breaking change to Event model breaks plugins | Plugins depend on stable Event fields (event_type, reel_targets, timestamps) — these are mature and stable |
| Over-engineering | Phase 1-3 are behavior-preserving refactors. Phase 4-5 are opt-in. No new deps. |

## Files Changed Summary

### New files
- `src/reel_plugins/__init__.py`
- `src/reel_plugins/base.py`
- `src/reel_plugins/registry.py`
- `src/reel_plugins/keeper.py`
- `src/reel_plugins/highlights.py`
- `src/reel_plugins/custom/` (Phase 5)
- `tests/unit/test_plugin_base.py`
- `tests/unit/test_keeper_plugin.py`
- `tests/unit/test_highlights_plugin.py`
- `tests/unit/test_plugin_registry.py`
- `tests/unit/test_plugin_clipper_integration.py`
- `tests/unit/test_plugin_regression.py`

### Modified files
- `src/api/worker.py` — replace hardcoded reel loop with plugin registry (Phase 3)
- `src/ingestion/models.py` — add `plugin_names` to Job (Phase 4)
- `src/api/routes/jobs.py` — accept plugin config (Phase 4)
- `src/config.py` — `DEFAULT_PLUGINS` env var (Phase 4)
- `tests/conftest.py` — add `make_plugin_event`, `make_context` fixtures

### Unchanged files
- `src/detection/base.py` — BaseDetector ABC stays as-is
- `src/detection/event_classifier.py` — PipelineRunner stays as-is
- `src/detection/goalkeeper_detector.py` — detection logic untouched
- `src/detection/models.py` — Event model unchanged (EVENT_REEL_MAP becomes docs-only)
- `src/segmentation/clipper.py` — compute_clips stays as-is (callers change)
- `src/assembly/composer.py` — ReelComposer stays as-is
