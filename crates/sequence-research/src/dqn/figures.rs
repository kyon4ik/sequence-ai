use std::path::Path;

use plotters::{coord::Shift, prelude::*};

use crate::Result;

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

#[derive(Clone, Copy)]
pub(super) enum Axis {
    Decisions,
    Episode,
    Checkpoint,
}

impl Axis {
    fn label(self) -> &'static str {
        match self {
            Self::Decisions => "Learner decisions",
            Self::Episode => "Training episode",
            Self::Checkpoint => "Checkpoint",
        }
    }

    fn tick(self, value: &usize) -> String {
        match (*value, self) {
            (1_000_000.., Self::Decisions) => format!("{:.2}M", *value as f64 / 1_000_000.0),
            (1_000.., Self::Decisions) => format!("{:.0}k", *value as f64 / 1_000.0),
            _ => value.to_string(),
        }
    }
}

#[derive(Default)]
pub(super) struct Curve {
    pub line: Vec<(usize, Option<f64>)>,
    pub samples: Vec<(usize, f64)>,
}

pub(super) struct Series<'a> {
    pub label: &'a str,
    pub panels: [Curve; 2],
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

pub(super) fn write_curves(
    path: &Path,
    title: &str,
    subtitle: &str,
    labels: [String; 2],
    axis: Axis,
    y_label: &str,
    series: &[Series<'_>],
) -> Result<()> {
    let footer_rows = series.len().div_ceil(4);
    let root = SVGBackend::new(path, (WIDTH, 460 + 28 * footer_rows as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    header(&root, title, subtitle)?;
    let (_, rest) = root.split_vertically(77);
    let (body, legend) = rest.split_vertically(367);
    let max_x = series
        .iter()
        .flat_map(|series| &series.panels)
        .flat_map(|panel| {
            panel
                .line
                .iter()
                .map(|&(x, _)| x)
                .chain(panel.samples.iter().map(|&(x, _)| x))
        })
        .max()
        .unwrap_or_default();
    let x_max = max_x.saturating_add((max_x / 40).max(1));

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
            .x_label_formatter(&|value| axis.tick(value))
            .y_label_formatter(&win_label)
            .x_desc(axis.label())
            .y_desc(y_label)
            .draw()?;

        for (index, series) in series.iter().enumerate() {
            let curve = &series.panels[panel];
            let accent = color(index);
            chart.draw_series(
                curve
                    .samples
                    .iter()
                    .map(|&point| Circle::new(point, 3, accent.mix(0.43).filled())),
            )?;
            for segment in curve.line.split(|&(_, value)| value.is_none()) {
                let points: Vec<_> = segment.iter().filter_map(|&(x, y)| Some((x, y?))).collect();
                if points.len() > 1 {
                    chart
                        .draw_series([PathElement::new(points.clone(), accent.stroke_width(2))])?;
                }
                chart.draw_series(
                    points
                        .into_iter()
                        .map(|point| Circle::new(point, 3, accent.filled())),
                )?;
            }
        }
    }
    for (index, series) in series.iter().enumerate() {
        let x = 30 + 302 * (index % 4) as i32;
        let y = 20 + 28 * (index / 4) as i32;
        let accent = color(index);
        legend.draw(&PathElement::new(
            [(x, y + 1), (x + 22, y + 1)],
            accent.stroke_width(3),
        ))?;
        legend.draw(&Text::new(
            series.label,
            (x + 32, y),
            (FONT, 15).into_font().color(&INK),
        ))?;
    }
    root.present()?;
    Ok(())
}
