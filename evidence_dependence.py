import json
from collections import Counter, defaultdict
from itertools import combinations

INPUT_FILE = "/opt/audit-lab/evidence_fusion.json"
OUTPUT_FILE = "/opt/audit-lab/evidence_dependence.json"


def load_data(filepath):
    with open(filepath) as file:
        return json.load(file)


def get_episodes(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("episodes", "results", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Unrecognized evidence_fusion.json structure")


def normalize_sources(evidence):
    sources = evidence.get("source_event_ids", [])

    if not isinstance(sources, list):
        return []

    return sorted({
        str(source)
        for source in sources
        if source is not None
    })


def analyze_episode(episode):
    episode_id = episode.get("episode_id", "UNKNOWN")
    evidence_records = episode.get("evidence", [])

    if not isinstance(evidence_records, list):
        evidence_records = []

    source_frequency = Counter()
    source_to_evidence = defaultdict(list)
    exact_source_groups = defaultdict(list)
    claim_groups = defaultdict(list)
    category_counts = Counter()
    signal_counts = Counter()
    pair_overlap_count = 0
    pair_count = 0

    records = []

    for item in evidence_records:
        if not isinstance(item, dict):
            continue

        evidence_id = item.get("evidence_id", "UNKNOWN")
        sources = normalize_sources(item)

        category = item.get("category", "UNKNOWN")
        signal = item.get("signal", "UNKNOWN")

        category_counts[category] += 1
        signal_counts[(category, signal)] += 1

        for source in sources:
            source_frequency[source] += 1
            source_to_evidence[source].append(evidence_id)

        if sources:
            exact_source_groups[tuple(sources)].append(evidence_id)

        claim_key = (
            category,
            signal,
            json.dumps(
                item.get("observed_value"),
                sort_keys=True,
                default=str
            ),
            json.dumps(
                item.get("baseline_value"),
                sort_keys=True,
                default=str
            ),
            json.dumps(
                sources,
                sort_keys=True
            )
        )
        claim_groups[claim_key].append(evidence_id)

        records.append({
            "evidence_id": evidence_id,
            "category": category,
            "signal": signal,
            "source_event_ids": sources,
            "provenance_scope": item.get("provenance_scope"),
            "interpretation": item.get("interpretation")
        })

    for first, second in combinations(records, 2):
        pair_count += 1

        sources1 = set(first["source_event_ids"])
        sources2 = set(second["source_event_ids"])

        if sources1.intersection(sources2):
            pair_overlap_count += 1

    duplicate_claim_groups = [
        {
            "evidence_ids": ids,
            "count": len(ids)
        }
        for ids in claim_groups.values()
        if len(ids) > 1
    ]

    shared_source_groups = [
        {
            "source_event_ids": list(sources),
            "evidence_ids": ids,
            "count": len(ids)
        }
        for sources, ids in exact_source_groups.items()
        if len(ids) > 1
    ]

    total_evidence = len(records)
    unique_sources = len(source_frequency)

    return {
        "episode_id": episode_id,
        "summary": {
            "evidence_record_count": total_evidence,
            "unique_source_event_count": unique_sources,
            "evidence_records_per_source_event": (
                round(total_evidence / unique_sources, 3)
                if unique_sources else None
            ),
            "evidence_pairs_compared": pair_count,
            "pairs_sharing_at_least_one_source": pair_overlap_count,
            "pairwise_source_overlap_fraction": (
                round(pair_overlap_count / pair_count, 4)
                if pair_count else None
            ),
            "category_counts": dict(category_counts),
            "signal_counts": {
                f"{category}::{signal}": count
                for (category, signal), count in signal_counts.items()
            }
        },
        "source_event_frequency": dict(source_frequency),
        "source_to_evidence": dict(source_to_evidence),
        "shared_source_groups": shared_source_groups,
        "potential_duplicate_claim_groups": duplicate_claim_groups,
        "evidence_records": records
    }


def main():
    data = load_data(INPUT_FILE)
    episodes = get_episodes(data)

    results = [
        analyze_episode(episode)
        for episode in episodes
        if isinstance(episode, dict)
    ]

    output = {
        "analysis_type": "evidence_source_overlap_diagnostic",
        "episode_count": len(results),
        "limitations": [
            "Shared source events indicate provenance overlap, not proof of statistical dependence.",
            "Potential duplicate claim groups require manual review.",
            "Distinct evidence records are not assumed to be independent confirmations.",
            "This diagnostic does not calculate maliciousness or threat scores.",
            "Missing source IDs can make overlap appear lower than it really is."
        ],
        "episodes": results
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(output, file, indent=2)

    print("[+] Episodes analyzed:", len(results))
    print("[+] Output:", OUTPUT_FILE)

    for episode in results:
        summary = episode["summary"]

        print(
            f"\n{episode['episode_id']}: "
            f"{summary['evidence_record_count']} evidence records, "
            f"{summary['unique_source_event_count']} unique source events, "
            f"{summary['pairs_sharing_at_least_one_source']}/"
            f"{summary['evidence_pairs_compared']} pairs overlap"
        )

        print(
            "    Potential duplicate claim groups:",
            len(episode["potential_duplicate_claim_groups"])
        )


if __name__ == "__main__":
    main()
