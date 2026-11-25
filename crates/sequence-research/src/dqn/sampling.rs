pub(super) fn targets(first: usize, last: usize, available: usize, percent: usize) -> Vec<usize> {
    let count = ((available as u128 * percent as u128).div_ceil(100) as usize)
        .max(2)
        .min(available);
    let span = last - first;
    let intervals = count.saturating_sub(1).min(span);
    if intervals == 0 {
        return vec![first];
    }

    (0..=intervals)
        .map(|i| {
            let offset = (span as u128 * i as u128 + (intervals / 2) as u128) / intervals as u128;
            first + offset as usize
        })
        .collect()
}

pub(super) fn nearest_index(
    positions: impl Iterator<Item = usize>,
    target: usize,
) -> Option<usize> {
    positions
        .enumerate()
        .min_by_key(|&(_, position)| (position.abs_diff(target), position))
        .map(|(index, _)| index)
}
