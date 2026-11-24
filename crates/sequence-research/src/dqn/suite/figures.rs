use std::path::Path;

use plotters::{coord::Shift, prelude::*};

use crate::Result;

use super::{OPPONENTS, Report, probe, statistics};

const INK: RGBColor = RGBColor(24, 38, 55);
const MUTED: RGBColor = RGBColor(77, 94, 111);
const GRID: RGBColor = RGBColor(224, 230, 236);
const PALETTE: [RGBColor; 4] = [
    RGBColor(38, 99, 170),
    RGBColor(180, 104, 26),
    RGBColor(21, 125, 103),
    RGBColor(145, 72, 149),
];
const FONT: &str = "sans-serif";
const WIDTH: u32 = 1240;

type Area<'a> = DrawingArea<SVGBackend<'a>, Shift>;

fn color(arm: usize) -> RGBColor {
    PALETTE.get(arm).copied().unwrap_or_else(|| {
        let (red, green, blue) = Palette99::pick(arm).rgb();
        RGBColor(red, green, blue)
    })
}

pub(super) fn write(directory: &Path, report: &Report<'_>) -> Result<()> {
    write_curves(
        &directory.join("win-rates.svg"),
        report,
        "Win rates across training",
        &format!(
            "{} shared validation deals · both seats · {} training seeds · small dots show individual seeds",
            report.config.validation.deal_count,
            report.config.seeds.len()
        ),
        OPPONENTS.map(|(_, label)| label.to_owned()),
        "Win rate (completed games)",
        |panel, run, id| report.validation[&(run, id)].matches.totals[panel].win_rate,
    )?;

    let report_names = ["Immediate win", "First sequence"];
    write_curves(
        &directory.join("tactics.svg"),
        report,
        "Tactical choices across training",
        "Fixed positions from Random vs Random games · small dots show individual seeds",
        std::array::from_fn(|panel| {
            let test = &report.validation.values().next().unwrap().tactics;
            let count = statistics::tactical_counts(test, probe::TACTICS[panel]).positions;
            format!("{} · {count} positions", report_names[panel])
        }),
        "Tactical action ranked first",
        |panel, run, id| {
            statistics::tactical_counts(
                &report.validation[&(run, id)].tactics,
                probe::TACTICS[panel],
            )
            .rate()
        },
    )?;
    write_endpoints(&directory.join("best-vs-last.svg"), report)
}

fn header(area: &Area<'_>, title: &str, subtitle: &str) -> Result<()> {
    area.draw(&Text::new(
        title,
        (25, 30),
        (FONT, 29).into_font().color(&INK),
    ))?;
    area.draw(&Text::new(
        subtitle,
        (26, 59),
        (FONT, 16).into_font().color(&MUTED),
    ))?;
    Ok(())
}

fn win_label(value: &f64) -> String {
    format!("{:.0}%", value * 100.0)
}

fn decision_label(decisions: &usize) -> String {
    match *decisions {
        0 => "0".to_owned(),
        1_000_000.. => format!("{:.2}M", *decisions as f64 / 1_000_000.0),
        1_000.. => format!("{:.0}k", *decisions as f64 / 1_000.0),
        _ => decisions.to_string(),
    }
}

fn write_curves(
    path: &Path,
    report: &Report<'_>,
    title: &str,
    subtitle: &str,
    labels: [String; 2],
    y_label: &str,
    rate: impl Fn(usize, usize, usize) -> Option<f64>,
) -> Result<()> {
    let footer_rows = report.config.arms.len().div_ceil(4);
    let root = SVGBackend::new(path, (WIDTH, 460 + 28 * footer_rows as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    header(&root, title, subtitle)?;
    let (_, rest) = root.split_vertically(77);
    let (body, legend) = rest.split_vertically(367);
    let max_decisions = report
        .runs
        .iter()
        .flat_map(|run| {
            run.sampled
                .iter()
                .flatten()
                .map(|&id| run.checkpoints[id].decisions)
        })
        .chain(report.targets.last().copied())
        .max()
        .unwrap();
    let x_max = max_decisions + (max_decisions / 40).max(1);

    for (panel, area) in body.split_evenly((1, 2)).iter().enumerate() {
        let mut chart = ChartBuilder::on(area)
            .caption(&labels[panel], (FONT, 20).into_font().color(&INK))
            .margin(8)
            .margin_right(25)
            .x_label_area_size(47)
            .y_label_area_size(66)
            .build_cartesian_2d(0..x_max, 0.0..1.0)?;
        chart
            .configure_mesh()
            .disable_x_mesh()
            .max_light_lines(1)
            .bold_line_style(GRID)
            .axis_style(MUTED.mix(0.6))
            .label_style((FONT, 14).into_font().color(&MUTED))
            .axis_desc_style((FONT, 15).into_font().color(&INK))
            .x_labels(7)
            .y_labels(6)
            .x_label_formatter(&decision_label)
            .y_label_formatter(&win_label)
            .x_desc("Learner decisions")
            .y_desc(y_label)
            .draw()?;

        for arm in 0..report.config.arms.len() {
            let seeds: Vec<_> = report
                .runs
                .iter()
                .enumerate()
                .filter(|(_, run)| run.arm == arm)
                .collect();
            let accent = color(arm);
            let mut means = Vec::new();
            for (point, &target) in report.targets.iter().enumerate() {
                let rates: Vec<_> = seeds
                    .iter()
                    .filter_map(|&(index, run)| {
                        let id = run.sampled[point]?;
                        Some((run.checkpoints[id].decisions, rate(panel, index, id)))
                    })
                    .collect();
                chart.draw_series(rates.iter().filter_map(|&(decisions, value)| {
                    Some(Circle::new(
                        (decisions, value?),
                        3,
                        accent.mix(0.43).filled(),
                    ))
                }))?;
                let mean = if rates.len() == seeds.len() {
                    statistics::complete_mean(rates.iter().map(|&(_, rate)| rate))
                } else {
                    None
                };
                means.push(mean.map(|mean| (target, mean)));
            }
            for segment in means.split(Option::is_none) {
                let points: Vec<_> = segment.iter().flatten().copied().collect();
                chart.draw_series([PathElement::new(points.clone(), accent.stroke_width(2))])?;
                chart.draw_series(
                    points
                        .into_iter()
                        .map(|point| Circle::new(point, 3, accent.filled())),
                )?;
            }
        }
    }
    for (arm, definition) in report.config.arms.iter().enumerate() {
        let x = 30 + 302 * (arm % 4) as i32;
        let y = 20 + 28 * (arm / 4) as i32;
        let accent = color(arm);
        legend.draw(&PathElement::new(
            [(x, y + 1), (x + 22, y + 1)],
            accent.stroke_width(3),
        ))?;
        legend.draw(&Text::new(
            definition.label.as_str(),
            (x + 32, y),
            (FONT, 15).into_font().color(&INK),
        ))?;
    }
    root.present()?;
    Ok(())
}

fn write_endpoints(path: &Path, report: &Report<'_>) -> Result<()> {
    let height = 220 + 68 * report.config.arms.len() as u32;
    let root = SVGBackend::new(path, (WIDTH, height)).into_drawing_area();
    root.fill(&WHITE)?;
    header(
        &root,
        "Best checkpoint vs last checkpoint",
        &format!(
            "{} fresh holdout deals · both seats · best checkpoint selected by validation win rate against Heuristic",
            report.config.holdout.deal_count,
        ),
    )?;
    let (_, rest) = root.split_vertically(77);
    let (body, legend) = rest.split_vertically(height - 130);
    let arms = report.config.arms.len();
    for (panel, area) in body.split_evenly((1, 2)).iter().enumerate() {
        let mut chart = ChartBuilder::on(area)
            .caption(OPPONENTS[panel].1, (FONT, 20).into_font().color(&INK))
            .margin(9)
            .margin_right(25)
            .x_label_area_size(44)
            .y_label_area_size(5)
            .build_cartesian_2d(0.0..1.0, 0..(arms as i32 * 10 + 1))?;
        chart
            .configure_mesh()
            .disable_y_axis()
            .disable_y_mesh()
            .max_light_lines(1)
            .axis_style(MUTED.mix(0.6))
            .bold_line_style(GRID)
            .label_style((FONT, 14).into_font().color(&MUTED))
            .x_labels(6)
            .x_label_formatter(&win_label)
            .x_desc("Win rate (completed games)")
            .draw()?;
        for (arm, definition) in report.config.arms.iter().enumerate() {
            let comparison = &report.comparisons[arm * OPPONENTS.len() + panel];
            let y = ((arms - arm) * 10) as i32;
            let accent = color(arm);
            let delta = match (comparison.best_minus_last, comparison.deal_paired_ci_95) {
                (Some(delta), Some([low, high])) => format!(
                    "{:+.1} pp [{:+.1}, {:+.1}]",
                    delta * 100.0,
                    low * 100.0,
                    high * 100.0
                ),
                (Some(delta), None) => format!("{:+.1} pp", delta * 100.0),
                _ => "Unavailable".to_owned(),
            };
            chart.draw_series([
                Text::new(
                    definition.label.clone(),
                    (0.01, y - 1),
                    (FONT, 15).into_font().color(&INK),
                ),
                Text::new(delta, (0.65, y - 1), (FONT, 14).into_font().color(&MUTED)),
            ])?;
            for (best, mean, row) in [
                (true, comparison.best_mean, y - 4),
                (false, comparison.last_mean, y - 7),
            ] {
                chart.draw_series(comparison.seeds.iter().filter_map(|seed| {
                    let rate = if best { seed.best? } else { seed.last? };
                    Some(Circle::new((rate, row), 3, accent.mix(0.5).filled()))
                }))?;
                if let Some(mean) = mean {
                    let glyph = if best {
                        accent.filled()
                    } else {
                        accent.stroke_width(2)
                    };
                    chart.draw_series([Circle::new((mean, row), 6, glyph)])?;
                }
            }
        }
    }
    legend.draw(&Circle::new((31, 20), 5, INK.filled()))?;
    legend.draw(&Text::new(
        "Best mean",
        (45, 20),
        (FONT, 15).into_font().color(&INK),
    ))?;
    legend.draw(&Circle::new((156, 20), 5, INK.stroke_width(2)))?;
    legend.draw(&Text::new(
        "Last mean",
        (170, 20),
        (FONT, 15).into_font().color(&INK),
    ))?;
    legend.draw(&Text::new("Small dots show seeds · labels show best minus last with 95% deal-paired bootstrap intervals", (280, 20), (FONT, 15).into_font().color(&MUTED)))?;
    root.present()?;
    Ok(())
}
