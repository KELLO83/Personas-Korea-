import re
import polars as pl


_AGE_GROUP_PATTERN = re.compile(r"^(\d{1,3})\s*대?$")


def normalize_age_group_tokens(raw_groups: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw_groups is None:
        return []

    if isinstance(raw_groups, str):
        tokens = [token.strip() for token in raw_groups.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in raw_groups if str(token).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        match = _AGE_GROUP_PATTERN.match(token)
        if not match:
            continue

        age = match.group(1)
        label = f"{int(age)}대"
        if label in seen:
            continue

        seen.add(label)
        normalized.append(label)

    return normalized


def _balanced_allocation(total: int, groups: list[str]) -> dict[str, int]:
    if total <= 0 or not groups:
        return {group: 0 for group in groups}

    base, remainder = divmod(total, len(groups))
    allocation = {group: base for group in groups}

    for group in reversed(groups):
        if remainder <= 0:
            break
        allocation[group] += 1
        remainder -= 1

    return allocation


def sample_age_groups(
    df: pl.DataFrame,
    *,
    age_groups: list[str],
    max_rows: int,
    random_seed: int = 42,
) -> pl.DataFrame:
    if max_rows <= 0:
        return df.filter(pl.lit(False))

    if "age_group" not in df.columns:
        return df.filter(pl.lit(False))

    df_filtered = df.filter(pl.col("age_group").is_in(age_groups))
    if df_filtered.is_empty():
        return df_filtered

    allocation = _balanced_allocation(total=max_rows, groups=age_groups)
    sampled_frames: list[pl.DataFrame] = []
    group_remainders: dict[str, pl.DataFrame] = {}
    kept_count = 0

    for age_group in age_groups:
        group_df = df_filtered.filter(pl.col("age_group") == age_group)
        if group_df.is_empty():
            continue

        shuffled = group_df.sample(fraction=1.0, shuffle=True, seed=random_seed)
        take = min(allocation.get(age_group, 0), group_df.height)
        
        sampled_frames.append(shuffled.head(take))
        group_remainders[age_group] = shuffled.slice(take)
        kept_count += take

    remaining_slots = max_rows - kept_count
    for age_group in age_groups:
        if remaining_slots <= 0:
            break
        
        rem = group_remainders.get(age_group)
        if rem is not None and not rem.is_empty():
            take = min(remaining_slots, rem.height)
            sampled_frames.append(rem.head(take))
            remaining_slots -= take

    if not sampled_frames:
        return df.filter(pl.lit(False))

    final_df = pl.concat(sampled_frames)
    if final_df.height <= 1:
        return final_df
    return final_df.sample(fraction=1.0, shuffle=True, seed=random_seed)
