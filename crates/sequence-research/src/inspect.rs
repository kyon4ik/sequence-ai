use std::{
    collections::BTreeMap,
    fs,
    io::{self, Cursor, Write},
    net::Ipv4Addr,
    path::Path,
    process::{Command, Stdio},
    thread,
};

use sequence_game::{Action, BOARD_SIZE, Board, Card, Player, Position, Rank, Space, Suit};
use sequence_match::{AppliedAction, Match};
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server};

use crate::{
    Result,
    experiment::{AgentKind, ExperimentConfig, GameEnd, GameRow},
};

pub fn serve(
    directory: &Path,
    game_id: Option<usize>,
    action: Option<usize>,
    no_open: bool,
) -> Result<()> {
    let run = Run::load(directory)?;
    let mut query = String::new();
    if let Some(id) = game_id {
        let game = run
            .games
            .get(&id)
            .ok_or_else(|| format!("game {id} does not exist"))?;
        query = format!("?game={id}");
        if let Some(number) = action {
            if number > game.actions {
                return Err(format!("game {id} has {} actions", game.actions).into());
            }
            query.push_str(&format!("&action={number}"));
        }
    }

    let catalog = serde_json::to_string(&run.catalog())?;
    let server = Server::http((Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| format!("could not start the replay viewer: {error}"))?;
    let address = server.server_addr().to_string();
    let url = format!("http://{address}/{query}");
    println!("Replay viewer at {url}");
    io::stdout().flush()?;
    if !no_open {
        thread::spawn(move || {
            if let Err(error) = open_browser(&url) {
                eprintln!("Could not open the browser: {error}. Open the printed address.");
            }
        });
    }
    for request in server.incoming_requests() {
        let response = serve_request(&run, &catalog, &address, &request);
        if let Err(error) = request.respond(response) {
            eprintln!("Could not send replay viewer response: {error}");
        }
    }
    Ok(())
}

fn open_browser(url: &str) -> io::Result<()> {
    let mut command = match std::env::consts::OS {
        "macos" => Command::new("open"),
        "windows" => Command::new("rundll32"),
        _ => Command::new("xdg-open"),
    };
    if cfg!(windows) {
        command.arg("url.dll,FileProtocolHandler");
    }
    let status = command
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "browser launcher exited with {status}"
        )))
    }
}

type HttpResponse = Response<Cursor<Vec<u8>>>;

fn serve_request(run: &Run, catalog: &str, address: &str, request: &Request) -> HttpResponse {
    if !request
        .headers()
        .iter()
        .any(|header| header.field.equiv("Host") && header.value.as_str() == address)
    {
        return response(403, "text/plain", "Use the address printed by the viewer.");
    }
    if request.method() != &Method::Get {
        return response(405, "text/plain", "Use GET.");
    }

    let path = request.url().split('?').next().unwrap_or_default();
    match path {
        "/" => response(200, "text/html", include_str!("inspect/index.html")),
        "/style.css" => response(200, "text/css", include_str!("inspect/style.css")),
        "/app.js" => response(200, "text/javascript", include_str!("inspect/app.js")),
        "/api/run" => response(200, "application/json", catalog),
        path => {
            let row = path
                .strip_prefix("/api/games/")
                .and_then(|id| id.parse().ok())
                .and_then(|id| run.games.get(&id));
            let Some(row) = row else {
                return json_error(404, "Game or resource not found.");
            };
            match run
                .replay(row)
                .and_then(|replay| Ok(serde_json::to_string(&replay)?))
            {
                Ok(body) => response(200, "application/json", body),
                Err(error) => json_error(422, &error.to_string()),
            }
        }
    }
}

fn response(status: u16, mime: &str, body: impl Into<String>) -> HttpResponse {
    Response::from_string(body)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", format!("{mime}; charset=utf-8")).unwrap())
        .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap())
}

fn json_error(status: u16, error: &str) -> HttpResponse {
    response(
        status,
        "application/json",
        serde_json::json!({ "error": error }).to_string(),
    )
}

struct Run {
    name: String,
    config: ExperimentConfig,
    games: BTreeMap<usize, GameRow>,
}

#[derive(Deserialize)]
struct RunMetadata {
    board_layout: Option<String>,
}

impl Run {
    fn load(directory: &Path) -> Result<Self> {
        let summary_path = directory.join("summary.json");
        let summary: RunMetadata = serde_json::from_reader(
            fs::File::open(&summary_path)
                .map_err(|error| format!("could not read {}: {error}", summary_path.display()))?,
        )
        .map_err(|error| format!("{}: {error}", summary_path.display()))?;
        if summary.board_layout.as_deref() != Some(Board::LAYOUT_ID) {
            return Err(format!(
                "Replaying {} requires the {} board layout. Run the experiment again.",
                directory.display(),
                Board::LAYOUT_ID
            )
            .into());
        }

        let config_path = directory.join("config.json");
        let config = serde_json::from_slice(
            &fs::read(&config_path)
                .map_err(|error| format!("could not read {}: {error}", config_path.display()))?,
        )
        .map_err(|error| format!("{}: {error}", config_path.display()))?;

        let games_path = directory.join("games.csv");
        let mut reader = csv::Reader::from_path(&games_path)
            .map_err(|error| format!("{}: {error}", games_path.display()))?;
        let mut games = BTreeMap::new();
        for row in reader.deserialize::<GameRow>() {
            let row = row.map_err(|error| format!("{}: {error}", games_path.display()))?;
            let id = row.game_id;
            if games.insert(id, row).is_some() {
                return Err(format!("{} has duplicate game ID {id}", games_path.display()).into());
            }
        }
        let name = directory.file_name().unwrap_or(directory.as_os_str());
        Ok(Self {
            name: name.to_string_lossy().into_owned(),
            config,
            games,
        })
    }

    fn catalog(&self) -> Catalog<'_> {
        Catalog {
            name: &self.name,
            policies: self.config.policies,
            games: self
                .games
                .values()
                .map(|row| CatalogGame {
                    game_id: row.game_id,
                    pair_id: row.pair_id,
                    shuffle_seed: row.shuffle_seed.to_string(),
                    policy_a_seat: Player::from(row.policy_a_seat).index(),
                    outcome: row.outcome,
                    actions: row.actions,
                })
                .collect(),
        }
    }

    fn replay(&self, row: &GameRow) -> Result<GameReplay> {
        let replay = row.reconstruct(&self.config)?;
        let mut frames = Vec::with_capacity(replay.actions.len() + 1);
        let mut board = Vec::new();
        replay.play(|_, game| {
            let frame = Frame::capture(game, &mut board);
            frames.push(frame);
        })?;
        Ok(GameReplay {
            board,
            actions: replay.actions.into_iter().map(Into::into).collect(),
            frames,
        })
    }
}

#[derive(Serialize)]
struct Catalog<'a> {
    name: &'a str,
    policies: [AgentKind; 2],
    games: Vec<CatalogGame>,
}

#[derive(Serialize)]
struct CatalogGame {
    game_id: usize,
    pair_id: usize,
    shuffle_seed: String,
    policy_a_seat: usize,
    outcome: GameEnd,
    actions: usize,
}

#[derive(Serialize)]
struct GameReplay {
    board: Vec<Option<String>>,
    actions: Vec<RecordedAction>,
    frames: Vec<Frame>,
}

#[derive(Serialize)]
struct RecordedAction {
    player: usize,
    #[serde(flatten)]
    action: ReplayAction,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReplayAction {
    Play { card: String, target: usize },
    Exchange { card: String },
    Pass,
}

impl From<AppliedAction> for RecordedAction {
    fn from(applied: AppliedAction) -> Self {
        let action = match applied.action {
            Action::Play { card, target } => ReplayAction::Play {
                card: card_label(card),
                target: cell_index(target),
            },
            Action::Exchange(card) => ReplayAction::Exchange {
                card: card_label(card),
            },
            Action::Pass => ReplayAction::Pass,
        };
        Self {
            player: applied.player.index(),
            action,
        }
    }
}

#[derive(Serialize)]
struct Frame {
    active_player: Option<usize>,
    chips: Vec<Option<usize>>,
    sequences: [Vec<Vec<usize>>; 2],
    hands: [Vec<HandCard>; 2],
}

impl Frame {
    fn capture(game: &Match, board: &mut Vec<Option<String>>) -> Self {
        let active = game.current_player();
        let views = [Player::One, Player::Two].map(|player| game.game().view(player));
        let one = &views[0];
        if board.is_empty() {
            *board = Position::all()
                .map(|position| match one.board.space(position) {
                    Space::Corner => None,
                    Space::Labeled { card, .. } => Some(card_label(card)),
                })
                .collect();
        }
        Self {
            active_player: active.map(Player::index),
            chips: Position::all()
                .map(|position| match one.board.space(position) {
                    Space::Corner => None,
                    Space::Labeled { chip, .. } => chip.map(Player::index),
                })
                .collect(),
            sequences: one.claimed_sequences.map(|sequences| {
                sequences
                    .iter()
                    .map(|sequence| sequence.spaces().iter().copied().map(cell_index).collect())
                    .collect()
            }),
            hands: views.map(|view| {
                let actions = if active == Some(view.player) {
                    &view.legal_actions[..]
                } else {
                    &[]
                };
                view.hand
                    .iter()
                    .map(|&card| HandCard::from_view(card, view.board, actions))
                    .collect()
            }),
        }
    }
}

#[derive(Serialize)]
struct HandCard {
    card: String,
    dead: bool,
    exchangeable: bool,
    targets: Vec<usize>,
}

impl HandCard {
    fn from_view(card: Card, board: &Board, actions: &[Action]) -> Self {
        let targets = actions
            .iter()
            .filter_map(|&action| match action {
                Action::Play {
                    card: played,
                    target,
                } if played == card => Some(cell_index(target)),
                _ => None,
            })
            .collect();
        Self {
            card: card_label(card),
            dead: board.is_dead_card(card),
            exchangeable: actions.contains(&Action::Exchange(card)),
            targets,
        }
    }
}

fn cell_index(position: Position) -> usize {
    position.row() * BOARD_SIZE + position.column()
}

fn card_label(card: Card) -> String {
    let rank = match card.rank {
        Rank::Ace => "A",
        Rank::Two => "2",
        Rank::Three => "3",
        Rank::Four => "4",
        Rank::Five => "5",
        Rank::Six => "6",
        Rank::Seven => "7",
        Rank::Eight => "8",
        Rank::Nine => "9",
        Rank::Ten => "10",
        Rank::Queen => "Q",
        Rank::King => "K",
        Rank::Jack => "J",
    };
    let suit = match card.suit {
        Suit::Clubs => '♣',
        Suit::Diamonds => '♦',
        Suit::Hearts => '♥',
        Suit::Spades => '♠',
    };
    format!("{rank}{suit}")
}
