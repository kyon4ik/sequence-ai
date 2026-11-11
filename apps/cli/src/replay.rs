use anyhow::{Context, Result};
use sequence_core::{Cell, Simple};
use sequence_runner::Replay;

use crate::artifacts::Store;

pub fn show(store: &Store, reference: &str, frame: Option<usize>, json: bool) -> Result<()> {
    let record = store.load_replay(reference)?;
    let mut replay = Replay::<Simple>::new(&record)
        .context("validating replay against the current game rules")?;
    let index = frame.unwrap_or(record.steps.len());
    replay.seek(index)?;
    if json {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &record)?;
        println!();
        return Ok(());
    }
    let snapshot = record.frame(index).unwrap();
    println!(
        "{} (Red) vs {} (Blue)",
        record.header.players[0].id, record.header.players[1].id
    );
    println!(
        "Frame {}/{} | turn {} | actor {} | {:?}",
        index,
        record.steps.len(),
        snapshot.turns,
        snapshot.current_player,
        record.termination
    );
    if index > 0 {
        let step = &record.steps[index - 1];
        println!(
            "{}: {} ({:.3} ms), reward {:?}",
            step.actor,
            step.action,
            step.decision_time.as_secs_f64() * 1000.,
            step.rewards
        );
    }
    println!("    a b c d e f g h i j");
    for (row, cells) in snapshot.board.cells.chunks(10).enumerate() {
        print!("{row:>2}  ");
        for cell in cells {
            print!(
                "{} ",
                match cell {
                    Cell::Empty => '.',
                    Cell::Corner => '*',
                    Cell::Occupied(sequence_core::Color::Red) => 'R',
                    Cell::Occupied(sequence_core::Color::Blue) => 'B',
                }
            );
        }
        println!();
    }
    println!(
        "Sequences R/B: {}/{} | actor hand: {:?} | opponent cards: {} | draw pile: {}",
        snapshot.board.sequences[0].len(),
        snapshot.board.sequences[1].len(),
        snapshot
            .hand
            .iter()
            .copied()
            .map(card_label)
            .collect::<Vec<_>>(),
        snapshot.opponent_hand_size,
        snapshot.draw_pile_size
    );
    println!("Use --frame N to inspect a position, or sequence serve for the browser explorer.");
    Ok(())
}

pub fn card_label(card: sequence_core::Card) -> String {
    let ranks = [
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
    ];
    let suits = ["♣", "♠", "♥", "♦"];
    format!("{}{}", ranks[card.id() >> 2], suits[card.id() & 3])
}
