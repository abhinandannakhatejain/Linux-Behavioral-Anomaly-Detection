import json
import math
from collections import defaultdict


FEATURES_FILE = "episode_features.json"
DEVIATIONS_FILE = "episode_deviations.json"
EPISODES_FILE = "episodes.json"
CORRELATIONS_FILE = "correlated_events_v2.json"
OUTPUT_FILE = "evidence_fusion.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, allow_nan=False)
        f.write("\n")


def validate_list(data, name):
    if not isinstance(data, list):
        raise ValueError(
            f"{name} must contain a JSON list."
        )


def index_by_episode(records, name):
    validate_list(records, name)

    result = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                f"{name} contains a non-object record."
            )

        episode_id = record.get("episode_id")

        if episode_id is None:
            continue

        if episode_id in result:
            raise ValueError(
                f"Duplicate episode_id {episode_id!r} in {name}."
            )

        result[episode_id] = record

    return result


def get_event_ids(episode):
    event_ids = episode.get("events", [])

    if not isinstance(event_ids, list):
        return set()

    return {
        event_id
        for event_id in event_ids
        if isinstance(event_id, str) and event_id
    }


def safe_get(data, *keys):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def add_evidence(
    evidence,
    episode_id,
    category,
    signal,
    observed_value,
    interpretation,
    source_event_ids=None,
    provenance_scope="event_specific",
    baseline_value=None,
    baseline_available=None,
    limitations=None
):
    valid_event_ids = {
        event_id
        for event_id in (source_event_ids or [])
        if isinstance(event_id, str) and event_id
    }

    evidence.append({
        "evidence_id": f"{episode_id}-EV-{len(evidence) + 1:04d}",
        "category": category,
        "signal": signal,
        "observed_value": observed_value,
        "baseline_value": baseline_value,
        "baseline_available": baseline_available,
        "interpretation": interpretation,
        "source_event_ids": sorted(valid_event_ids),
        "provenance_scope": provenance_scope,
        "limitations": limitations or []
    })


def extract_identity_evidence(feature, episode, evidence, episode_id):
    identity = feature.get("identity", {})

    if not isinstance(identity, dict):
        return

    episode_event_ids = get_event_ids(episode)

    if identity.get("privilege_transition") is True:
        add_evidence(
            evidence,
            episode_id,
            "identity",
            "PRIVILEGE_TRANSITION",
            {
                "uid": identity.get("uid"),
                "euid": identity.get("euid"),
                "auid": identity.get("auid")
            },
            (
                "The episode records a privilege-context transition "
                "or a UID/EUID mismatch."
            ),
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            limitations=[
                "The evidence is associated with the episode, "
                "not a specific transition event.",
                "A privilege transition is not inherently malicious.",
                "This feature alone does not establish authorization or intent."
            ]
        )

    if identity.get("root_context") is True:
        add_evidence(
            evidence,
            episode_id,
            "identity",
            "ROOT_CONTEXT",
            {
                "uid": identity.get("uid"),
                "euid": identity.get("euid")
            },
            "The episode includes execution in a root-privileged context.",
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            limitations=[
                "The evidence is associated with the episode, "
                "not a specific event.",
                "Root context alone is not evidence of malicious activity."
            ]
        )


def build_behavior_event_index(episode):
    event_ids_by_behavior = defaultdict(list)
    behavior_sequence = episode.get("behavior_sequence", [])

    if not isinstance(behavior_sequence, list):
        return event_ids_by_behavior

    for item in behavior_sequence:
        if not isinstance(item, dict):
            continue

        event_id = item.get("event_id")

        if not isinstance(event_id, str) or not event_id:
            continue

        behaviors = item.get("behaviors", [])

        if not isinstance(behaviors, list):
            behaviors = []

        # Support records that expose a single behavior instead.
        single_behavior = item.get("behavior")

        if isinstance(single_behavior, str):
            behaviors = list(behaviors) + [single_behavior]

        for behavior in set(
            value for value in behaviors
            if isinstance(value, str) and value
        ):
            event_ids_by_behavior[behavior].append(event_id)

    return event_ids_by_behavior


def extract_behavior_evidence(episode, deviation, evidence, episode_id):
    if not isinstance(deviation, dict):
        return

    observed_behaviors = safe_get(
        deviation, "behavioral", "observed_behaviors"
    )

    novel_behaviors = safe_get(
        deviation, "behavioral", "novel_behaviors"
    )

    primary_behaviors = safe_get(
        deviation, "behavioral", "primary_behaviors"
    )

    if not isinstance(observed_behaviors, dict):
        observed_behaviors = {}

    if not isinstance(novel_behaviors, list):
        novel_behaviors = []

    if not isinstance(primary_behaviors, dict):
        primary_behaviors = {}

    baseline_available = safe_get(
        deviation, "behavioral", "profile_available"
    )

    event_ids_by_behavior = build_behavior_event_index(episode)

    for behavior, details in observed_behaviors.items():
        if not isinstance(behavior, str):
            continue

        if not isinstance(details, dict):
            details = {"value": details}

        add_evidence(
            evidence,
            episode_id,
            "behavioral",
            behavior,
            details,
            "Behavior was observed and summarized by deviation analysis.",
            source_event_ids=event_ids_by_behavior.get(behavior, []),
            baseline_value=details.get("profile_frequency"),
            baseline_available=baseline_available,
            limitations=[
                "Behavior labels are derived from telemetry and detection rules.",
                "A behavior label does not independently establish malicious intent.",
                "Behavior evidence may overlap with other evidence categories."
            ]
        )

    for behavior in novel_behaviors:
        if not isinstance(behavior, str):
            continue

        add_evidence(
            evidence,
            episode_id,
            "behavioral_novelty",
            "NOVEL_BEHAVIOR",
            behavior,
            "The deviation analysis marked this behavior as novel.",
            source_event_ids=event_ids_by_behavior.get(behavior, []),
            baseline_available=baseline_available,
            limitations=[
                "Novelty means absent from the modeled profile, not malicious.",
                "The reliability of novelty depends on baseline quality."
            ]
        )

    for behavior, details in primary_behaviors.items():
        if not isinstance(behavior, str):
            continue

        if not isinstance(details, dict):
            details = {}

        add_evidence(
            evidence,
            episode_id,
            "behavioral_frequency",
            "PRIMARY_BEHAVIOR_FREQUENCY",
            {
                "behavior": behavior,
                "current_count": details.get("current_count"),
                "profile_event_rate": details.get("profile_event_rate")
            },
            "The behavior's event count or rate was summarized by deviation analysis.",
            source_event_ids=event_ids_by_behavior.get(behavior, []),
            baseline_available=details.get("profile_available"),
            limitations=[
                "Event counts can be affected by audit record multiplicity.",
                "This signal may overlap with other behavior evidence."
            ]
        )


def extract_deviation_evidence(episode, deviation, evidence, episode_id):
    episode_event_ids = get_event_ids(episode)

    if not isinstance(deviation, dict):
        add_evidence(
            evidence,
            episode_id,
            "data_quality",
            "DEVIATION_RECORD_MISSING",
            None,
            "No deviation record was found for this episode.",
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            baseline_available=False,
            limitations=[
                "Behavioral deviation could not be assessed.",
                "Missing deviation data must not be interpreted as normal behavior."
            ]
        )
        return

    baseline_available = deviation.get(
        "baseline_available", False
    ) is True

    if not baseline_available:
        add_evidence(
            evidence,
            episode_id,
            "data_quality",
            "BASELINE_UNAVAILABLE",
            {
                "profile_used": deviation.get("profile_used"),
                "profile_episode_count": deviation.get(
                    "profile_episode_count"
                )
            },
            "A usable baseline was unavailable for this episode.",
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            baseline_available=False,
            limitations=[
                deviation.get(
                    "reason",
                    "Insufficient baseline data."
                ),
                "Unavailable baseline information is not evidence of normality."
            ]
        )

    # Temporal metrics are processed independently of contextual novelty.
    # A metric is marked baseline-available only when the overall baseline
    # is available and the metric contains a usable comparison.
    temporal = deviation.get("temporal", {})

    if not isinstance(temporal, dict):
        temporal = {}

    for metric, details in temporal.items():
        if not isinstance(details, dict):
            continue

        z_score = details.get("z_score")
        value = details.get("value")

        if z_score is None and value is None:
            continue

        metric_baseline_available = (
            baseline_available
            and finite_number(z_score)
        )

        baseline_value = {
            "mean": details.get("mean"),
            "std": details.get("std"),
            "min": details.get("min"),
            "max": details.get("max")
        }

        add_evidence(
            evidence,
            episode_id,
            "temporal_deviation",
            "TEMPORAL_METRIC",
            {
                "metric": metric,
                "value": value,
                "z_score": z_score,
                "out_of_range": details.get("out_of_range")
            },
            (
                "Temporal metric was compared with the episode profile."
                if metric_baseline_available
                else "Temporal metric was reported, but a usable baseline comparison "
                     "was not established."
            ),
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            baseline_value=(
                baseline_value if metric_baseline_available else None
            ),
            baseline_available=metric_baseline_available,
            limitations=[
                "The profile contains few episodes.",
                "Temporal metrics may be correlated with one another.",
                "A deviation is not proof of malicious activity."
            ]
        )

    process = deviation.get("process", {})

    if not isinstance(process, dict):
        process = {}

    numerical = process.get("numerical", {})

    if not isinstance(numerical, dict):
        numerical = {}

    for metric, details in numerical.items():
        if not isinstance(details, dict):
            continue

        z_score = details.get("z_score")
        value = details.get("value")

        if z_score is None and value is None:
            continue

        metric_baseline_available = (
            baseline_available
            and finite_number(z_score)
        )

        baseline_value = {
            "mean": details.get("mean"),
            "std": details.get("std"),
            "min": details.get("min"),
            "max": details.get("max")
        }

        add_evidence(
            evidence,
            episode_id,
            "process_deviation",
            "PROCESS_METRIC",
            {
                "metric": metric,
                "value": value,
                "z_score": z_score,
                "out_of_range": details.get("out_of_range")
            },
            (
                "Process-structure metric was compared with the profile."
                if metric_baseline_available
                else "Process metric was reported, but a usable baseline comparison "
                     "was not established."
            ),
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            baseline_value=(
                baseline_value if metric_baseline_available else None
            ),
            baseline_available=metric_baseline_available,
            limitations=[
                "Small baseline size limits reliability.",
                "Process metrics may describe the same underlying activity."
            ]
        )

     # Contextual novelty uses item-level provenance from deviation records.
    # Each novel target, command, or syscall should reference only the events
    # that support that specific novelty claim.
    context = deviation.get("context", {})

    if not isinstance(context, dict):
        context = {}

    context_categories = (
        ("novel_targets", "targets", "target"),
        ("novel_commands", "commands", "command_line"),
        ("novel_syscalls", "syscalls", "syscall")
    )

    for category, record_category, value_key in context_categories:
        values = context.get(category, [])

        if not isinstance(values, list):
            continue

        for value in values:
            source_event_ids = []
            provenance_scope = "item_level"

            # Deviation records may be dictionaries containing their own
            # source_event_ids. Prefer those precise references.
            if isinstance(value, dict):
                item_value = value.get(
                    value_key,
                    value.get("value", value.get("item", str(value)))
                )

                candidate_ids = value.get("source_event_ids", [])
                if isinstance(candidate_ids, list):
                    source_event_ids = candidate_ids

            else:
                item_value = value

                # Support the current deviation output, where novel_* lists
                # may contain strings and detailed records live separately.
                detailed_records = context.get(record_category, [])
                if isinstance(detailed_records, list):
                    for record in detailed_records:
                        if not isinstance(record, dict):
                            continue

                        if record.get(value_key) == value and record.get("novel"):
                            candidate_ids = record.get("source_event_ids", [])
                            if isinstance(candidate_ids, list):
                                source_event_ids.extend(candidate_ids)

                # Remove repeated IDs while preserving their original order.
                source_event_ids = list(dict.fromkeys(source_event_ids))

            # Never silently claim item-level provenance if no supporting
            # source IDs were found.
            if not source_event_ids:
                provenance_scope = "episode_level"
                source_event_ids = episode_event_ids

            add_evidence(
                evidence,
                episode_id,
                "contextual_novelty",
                category.upper(),
                item_value,
                (
                    "The contextual item was marked novel by profile comparison."
                    if baseline_available
                    else "The deviation record contains a contextual novelty item, "
                         "but the overall baseline is unavailable."
                ),
                source_event_ids=source_event_ids,
                provenance_scope=provenance_scope,
                baseline_available=baseline_available,
                limitations=[
                    "Novelty alone does not imply maliciousness.",
                    "Missing item-level provenance falls back to episode-level "
                    "event references.",
                    "Verify the source of the novelty decision when the baseline "
                    "is unavailable."
                ]
            )


def extract_correlation_evidence(episode, correlations, evidence, episode_id):
    episode_event_ids = get_event_ids(episode)

    for source_index, correlation in enumerate(correlations):
        if not isinstance(correlation, dict):
            continue

        event1 = correlation.get("event1")
        event2 = correlation.get("event2")

        valid_event1 = isinstance(event1, str) and bool(event1)
        valid_event2 = isinstance(event2, str) and bool(event2)

        if not valid_event1 and not valid_event2:
            continue

        event1_in_episode = (
            valid_event1 and event1 in episode_event_ids
        )
        event2_in_episode = (
            valid_event2 and event2 in episode_event_ids
        )

        matched_ids = {
            event_id
            for event_id, is_member in (
                (event1, event1_in_episode),
                (event2, event2_in_episode)
            )
            if is_member
        }

        if not matched_ids:
            continue

        if event1_in_episode and event2_in_episode:
            correlation_scope = "internal"
        else:
            correlation_scope = "boundary"

        external_ids = sorted({
            event_id
            for event_id, is_member in (
                (event1, event1_in_episode),
                (event2, event2_in_episode)
            )
            if isinstance(event_id, str)
            and event_id
            and not is_member
        })

        # Stable reference to the source-list position.
        # The same source correlation keeps the same reference across episodes.
        correlation_ref = f"CORR-{source_index + 1:04d}"

        observed_value = {
            "correlation_ref": correlation_ref,
            "source_record_index": source_index,
            "scope": correlation_scope,
            "event1": event1,
            "event2": event2,
            "event1_in_episode": event1_in_episode,
            "event2_in_episode": event2_in_episode,
            "external_event_ids": external_ids,
            "behavior1": correlation.get("behavior1"),
            "behavior2": correlation.get("behavior2"),
            "relationship": correlation.get("relationship"),
            "time_difference": correlation.get("time_difference"),
            "ancestry_distance": correlation.get("ancestry_distance"),
            "shared_targets": correlation.get("shared_targets", [])
        }

        limitations = [
            "A correlation describes a relationship between events, not proof of causation.",
            "Correlated events must not be assumed to be statistically independent evidence.",
            "The source-list reference depends on the ordering of the correlation input file."
        ]

        if correlation_scope == "boundary":
            limitations.append(
                "At least one endpoint is outside this episode; external endpoints are context, not episode members."
            )

        add_evidence(
            evidence,
            episode_id,
            "event_correlation",
            "CORRELATED_EVENTS",
            observed_value,
            (
                f"A source correlation ({correlation_ref}) references "
                f"this episode with {correlation_scope} scope."
            ),
            source_event_ids=matched_ids,
            provenance_scope=(
                "episode_internal_correlation"
                if correlation_scope == "internal"
                else "episode_with_external_context"
            ),
            limitations=limitations
        )

def build_episode_record(
    episode,
    feature,
    deviation,
    correlations
):
    episode_id = episode.get("episode_id")
    evidence = []

    episode_event_ids = sorted(get_event_ids(episode))

    if not feature:
        add_evidence(
            evidence,
            episode_id,
            "data_quality",
            "FEATURE_RECORD_MISSING",
            None,
            "No feature record was found for this episode.",
            source_event_ids=episode_event_ids,
            provenance_scope="episode_level",
            baseline_available=False,
            limitations=[
                "Identity and feature-derived evidence may be incomplete."
            ]
        )
    else:
        extract_identity_evidence(feature, episode, evidence, episode_id)

    extract_behavior_evidence(episode, deviation, evidence, episode_id)

    extract_deviation_evidence(
        episode,
        deviation,
        evidence,
        episode_id
    )

    extract_correlation_evidence(
        episode,
        correlations,
        evidence,
        episode_id
    )

    return {
        "episode_id": episode.get("episode_id"),
        "profile_used": (
            deviation.get("profile_used")
            if isinstance(deviation, dict)
            else None
        ),
        "profile_available": (
            deviation.get("profile_available", False)
            if isinstance(deviation, dict)
            else False
        ),
        "baseline_available": (
            deviation.get("baseline_available", False)
            if isinstance(deviation, dict)
            else False
        ),
        "baseline_reliable": (
            deviation.get("baseline_reliable", False)
            if isinstance(deviation, dict)
            else False
        ),
        "reference_episode_count": (
            deviation.get("reference_episode_count", 0)
            if isinstance(deviation, dict)
            else 0
        ),
        "event_count": len(episode_event_ids),
        "source_event_ids": episode_event_ids,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "fusion_notes": [
            "Evidence is organized, not converted into a threat score.",
            "Evidence records may share underlying source events.",
            "Correlated signals must not be assumed statistically independent.",
            "Episode-level provenance does not identify a specific causal event.",
            "External correlation endpoints are context, not episode members."
        ]
    }


def main():
    features = load_json(FEATURES_FILE)
    deviations_data = load_json(DEVIATIONS_FILE)
    episodes = load_json(EPISODES_FILE)
    correlations = load_json(CORRELATIONS_FILE)

    validate_list(episodes, EPISODES_FILE)
    validate_list(correlations, CORRELATIONS_FILE)

    if not isinstance(deviations_data, dict):
        raise ValueError(
            f"{DEVIATIONS_FILE} must contain a JSON object."
        )

    deviations = index_by_episode(
        deviations_data.get("deviations", []),
        DEVIATIONS_FILE
    )

    feature_index = index_by_episode(
        features,
        FEATURES_FILE
    )

    # Reject duplicate episode IDs in the main episode list.
    seen_episode_ids = set()

    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError(
                f"{EPISODES_FILE} contains a non-object episode."
            )

        episode_id = episode.get("episode_id")

        if episode_id is None:
            raise ValueError(
                f"An episode in {EPISODES_FILE} has no episode_id."
            )

        if episode_id in seen_episode_ids:
            raise ValueError(
                f"Duplicate episode_id {episode_id!r} in {EPISODES_FILE}."
            )

        seen_episode_ids.add(episode_id)

    fused_records = []

    for episode in episodes:
        episode_id = episode.get("episode_id")

        feature = feature_index.get(episode_id)
        deviation = deviations.get(episode_id)

        record = build_episode_record(
            episode,
            feature,
            deviation,
            correlations
        )

        fused_records.append(record)

    output = {
        "fusion_version": 2,
        "source_files": {
            "features": FEATURES_FILE,
            "deviations": DEVIATIONS_FILE,
            "episodes": EPISODES_FILE,
            "correlations": CORRELATIONS_FILE
        },
        "episode_count": len(fused_records),
        "evidence_record_count": sum(
            record["evidence_count"]
            for record in fused_records
        ),
        "episodes": fused_records,
        "limitations": [
            "This output is not a calibrated threat score.",
            "The current behavioral baseline is very small.",
            "Evidence records can share source events.",
            "Audit telemetry may not capture every relevant action.",
            "Correlation does not establish causation or malicious intent.",
            "Episode-level provenance does not prove that every event in an "
            "episode independently supports the associated evidence.",
            "Contextual novelty recorded without a usable baseline requires "
            "additional verification."
        ]
    }

    save_json(OUTPUT_FILE, output)

    print(f"Processed episodes: {len(fused_records)}")
    print(
        "Evidence records:",
        output["evidence_record_count"]
    )
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

