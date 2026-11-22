use std::path::Path;

use plotters::prelude::*;
use sequence_match::EvaluationReport;

use crate::{Result, summary::Summary};

const SIZE: (u32, u32) = (800, 420);
const INK: RGBColor = RGBColor(30, 42, 55);
const ACCENT: RGBColor = RGBColor(23, 99, 121);
const GRID: RGBColor = RGBColor(223, 230, 234);
const FONT: &str = "sans-serif";

pub fn write_figures(directory: &Path, report: &EvaluationReport, summary: &Summary) -> Result<()> {
    write_score(&directory.join("match-score.svg"), summary)?;
    write_lengths(&directory.join("game-lengths.svg"), report, summary)
}

fn write_score(path: &Path, summary: &Summary) -> Result<()> {
    let rows = [
        (3, &summary.overall, summary.paired_ci_95),
        (2, &summary.starts, None),
        (1, &summary.moves_second, None),
    ]
    .map(|(y, score, ci)| {
        let value = score.score.map(|value| value * 100.0);
        let interval = ci
            .or_else(|| score.score.is_none().then_some(score.score_bounds))
            .map(|bounds| bounds.map(|value| value * 100.0));
        (y, value, interval)
    });
    let mut minimum = 100.0_f64;
    let mut maximum = 0.0_f64;
    for (_, value, interval) in rows {
        for percent in value.into_iter().chain(interval.into_iter().flatten()) {
            minimum = minimum.min(percent);
            maximum = maximum.max(percent);
        }
    }

    let padding = ((maximum - minimum) * 0.15).max(0.5);
    let left = (minimum - padding).floor().max(0.0);
    let right = (maximum + padding).ceil().min(100.0);

    let root = SVGBackend::new(path, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let [policy_a, policy_b] = summary.policies;
    let (body, footer) = root.split_vertically(380);
    let mut chart = ChartBuilder::on(&body)
        .caption(
            format!("{policy_a} (A) vs {policy_b} (B)"),
            (FONT, 28).into_font().color(&INK),
        )
        .margin(18)
        .x_label_area_size(56)
        .y_label_area_size(125)
        .build_cartesian_2d(left..right, (0..4).with_key_points(vec![1, 2, 3]))?;
    chart
        .configure_mesh()
        .disable_y_mesh()
        .max_light_lines(1)
        .light_line_style(GRID)
        .bold_line_style(GRID)
        .label_style((FONT, 17).into_font().color(&INK))
        .axis_desc_style((FONT, 19).into_font().color(&INK))
        .y_label_formatter(&|y| match y {
            3 => "Both seats".into(),
            2 => "Starts".into(),
            1 => "Moves second".into(),
            _ => String::new(),
        })
        .x_labels(6)
        .x_label_formatter(&|x| format!("{x:.1}%"))
        .x_desc("Match score for policy A")
        .draw()?;

    for (y, value, interval) in rows {
        if let Some([lower, upper]) = interval {
            chart.draw_series([PathElement::new(
                [(lower, y), (upper, y)],
                ACCENT.stroke_width(5),
            )])?;
        }
        if let Some(value) = value {
            chart.draw_series([Circle::new((value, y), 6, ACCENT.filled())])?;
        }
    }

    let note = match (summary.paired_ci_95, summary.overall.score) {
        (Some(_), _) => "Overall bar shows the 95% paired bootstrap interval",
        (None, Some(_)) => "A paired bootstrap interval requires at least two deals",
        (None, None) => "Bars show score bounds when games reach the action limit",
    };
    footer.draw(&Text::new(
        format!("{} paired deals  |  {note}", summary.deals),
        (24, 17),
        (FONT, 16).into_font().color(&INK),
    ))?;
    root.present()?;
    Ok(())
}

fn write_lengths(path: &Path, report: &EvaluationReport, summary: &Summary) -> Result<()> {
    let action_counts = || report.pairs.iter().flatten().map(|game| game.action_count);
    let min = action_counts().min().expect("at least one deal");
    let max = action_counts().max().expect("at least one deal");
    let width = (max - min + 1).div_ceil(20);
    let mut bins = vec![0usize; (max - min) / width + 1];
    for count in action_counts() {
        bins[(count - min) / width] += 1;
    }

    let root = SVGBackend::new(path, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let (body, footer) = root.split_vertically(380);
    let min_edge = min as f64 - 0.5;
    let max_edge = min_edge + (width * bins.len()) as f64;
    let peak = bins.iter().copied().max().unwrap();
    let [policy_a, policy_b] = summary.policies;
    let mut chart = ChartBuilder::on(&body)
        .caption(
            format!("Game lengths for {policy_a} vs {policy_b}"),
            (FONT, 28).into_font().color(&INK),
        )
        .margin(18)
        .x_label_area_size(56)
        .y_label_area_size(64)
        .build_cartesian_2d(min_edge..max_edge, 0..peak + peak.div_ceil(10))?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .max_light_lines(1)
        .light_line_style(GRID)
        .bold_line_style(GRID)
        .label_style((FONT, 17).into_font().color(&INK))
        .axis_desc_style((FONT, 19).into_font().color(&INK))
        .x_labels(if min == max { 1 } else { 8 })
        .x_label_formatter(&|x| format!("{x:.0}"))
        .x_desc("Accepted actions per game")
        .y_desc("Games")
        .draw()?;
    chart.draw_series(bins.iter().enumerate().map(|(index, &count)| {
        let left = min_edge + (index * width) as f64;
        let right = left + width as f64;
        let gap = width as f64 * 0.06;
        Rectangle::new([(left + gap, 0), (right - gap, count)], ACCENT.filled())
    }))?;
    chart.draw_series([PathElement::new(
        [(summary.median_actions, 0), (summary.median_actions, peak)],
        INK.stroke_width(3),
    )])?;

    footer.draw(&Text::new(
        format!(
            "Median {} actions  |  {} games, including {} at the action limit",
            summary.median_actions,
            summary.deals * 2,
            summary.overall.counts.truncations,
        ),
        (24, 17),
        (FONT, 16).into_font().color(&INK),
    ))?;
    root.present()?;
    Ok(())
}
