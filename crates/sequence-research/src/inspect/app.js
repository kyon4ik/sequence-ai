const byId = (id) => document.getElementById(id);
const filters = {
  perspective: byId("perspective"),
  moveOrder: byId("move-order"),
  search: byId("search"),
  sort: byId("sort"),
};
const outcomes = {
  all: ["All", "All"],
  win: ["Wins", "Win"],
  loss: ["Losses", "Loss"],
  draw: ["Draws", "Draw"],
  action_limit: ["Action limit", "Action limit"],
};
const seats = ["player-one", "player-two"];
const suits = { "♣": "clubs", "♦": "diamonds", "♥": "hearts", "♠": "spades" };
const ranks = { A: "Ace", J: "Jack", Q: "Queen", K: "King" };
const columns = "ABCDEFGHIJ";
const numberFormat = new Intl.NumberFormat();
const gamesPerPage = 100;
const gameRows = byId("game-rows");
const replayMessage = byId("replay-message");
const replayContent = byId("replay-content");
const scrubber = byId("scrubber");
const actionsList = byId("actions");
const hands = byId("hands");

let run;
let outcomeFilter = "all";
let catalogPage = 0;
let game;
let replay;
let frameIndex = 0;
let selectedCard = null;
let pendingRequest;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(className, text) {
  const node = element("button", className, text);
  node.type = "button";
  return node;
}

async function getJson(path, signal) {
  const response = await fetch(path, { signal });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The viewer could not load the data.");
  return body;
}

function policyLabel(policy) {
  return `${["A", "B"][policy]} · ${run.policies[policy]}`;
}

function seatLabel(row, seat) {
  const policy = row.policy_a_seat === seat ? 0 : 1;
  return policyLabel(policy);
}

function perspectiveSeat(row) {
  return filters.perspective.value === "0" ? row.policy_a_seat : 1 - row.policy_a_seat;
}

function resultFor(row) {
  if (row.outcome === "draw" || row.outcome === "action_limit") return row.outcome;
  const winner = row.outcome === "player_one_win" ? 0 : 1;
  return winner === perspectiveSeat(row) ? "win" : "loss";
}

function renderCatalog() {
  const search = filters.search.value.trim();
  const counts = { all: 0, win: 0, loss: 0, draw: 0, action_limit: 0 };
  const visible = [];
  for (const row of run.games) {
    const first = perspectiveSeat(row) === run.starting_seat;
    const order = first ? "first" : "second";
    if (filters.moveOrder.value !== "all" && filters.moveOrder.value !== order) continue;
    if (search && search !== String(row.game_id) && search !== row.shuffle_seed) continue;

    const result = resultFor(row);
    counts[result]++;
    counts.all++;
    if (outcomeFilter === "all" || outcomeFilter === result) visible.push({ row, order, result });
  }

  if (filters.sort.value === "shortest") {
    visible.sort((a, b) => a.row.actions - b.row.actions);
  } else if (filters.sort.value === "longest") {
    visible.sort((a, b) => b.row.actions - a.row.actions);
  }

  for (const option of byId("outcomes").children) {
    const key = option.dataset.result;
    option.lastElementChild.textContent = " " + numberFormat.format(counts[key]);
    option.setAttribute("aria-pressed", String(key === outcomeFilter));
  }

  const pageStart = catalogPage * gamesPerPage;
  const page = visible.slice(pageStart, pageStart + gamesPerPage);
  gameRows.replaceChildren(...page.map(({ row, order, result }) => {
    const tr = element("tr");
    tr.dataset.game = row.game_id;
    const link = button("game-link", `#${row.game_id}`);
    link.setAttribute("aria-label", `Open game ${row.game_id}`);
    const idCell = element("td");
    idCell.append(link);
    tr.append(
      idCell,
      element("td", `result-${result}`, outcomes[result][1]),
      element("td", "", order === "first" ? "First" : "Second"),
      element("td", "", row.actions),
    );
    return tr;
  }));
  gameRows.closest(".games-table").scrollTop = 0;
  markSelectedGame();
  const matches = numberFormat.format(visible.length);
  const total = numberFormat.format(run.games.length);
  const paginated = visible.length > gamesPerPage;
  const range = `${numberFormat.format(pageStart + 1)}–${numberFormat.format(pageStart + page.length)}`;
  byId("result-count").textContent = !visible.length
    ? "No games match these filters."
    : paginated ? `${range} of ${matches} matching games` : `${matches} of ${total} games`;
  byId("pages").hidden = !paginated;
  byId("previous-page").disabled = catalogPage === 0;
  byId("next-page").disabled = pageStart + page.length >= visible.length;
  if (game && replay) renderGameResult();
}

function applyFilters() {
  catalogPage = 0;
  renderCatalog();
}

function markSelectedGame() {
  for (const row of gameRows.children) {
    const selected = row.dataset.game === String(game?.game_id);
    row.classList.toggle("selected", selected);
    row.querySelector("button").setAttribute("aria-current", String(selected));
  }
}

function renderGameResult() {
  const result = resultFor(game);
  const label = byId("game-result");
  label.className = `result-${result}`;
  label.textContent = outcomes[result][1];
}

function cardFace(card) {
  const suit = card.at(-1);
  return element("span", ["♦", "♥"].includes(suit) ? "red-suit" : "", card);
}

function cardName(card) {
  const rank = card.slice(0, -1);
  return `${ranks[rank] || rank} of ${suits[card.at(-1)]}`;
}

function coordinate(index) {
  return `${columns[index % 10]}${Math.floor(index / 10) + 1}`;
}

function actionText(action, frame) {
  switch (action.kind) {
    case "play":
      return frame.chips[action.target] === null
        ? `Play ${action.card} at ${coordinate(action.target)}`
        : `Remove ${coordinate(action.target)} with ${action.card}`;
    case "exchange":
      return `Exchange ${action.card}`;
    case "pass":
      return "Pass";
  }
}

function renderPlayers(frame) {
  byId("players").replaceChildren(...seats.map((seat, index) => {
    const player = element("span", "player");
    player.classList.toggle("active", index === frame.active_player);
    const count = frame.sequences[index].length;
    player.append(
      element("span", `player-dot ${seat}`),
      `${seatLabel(game, index)} · ${count} ${count === 1 ? "sequence" : "sequences"}`,
    );
    return player;
  }));
}

function renderBoard(frame, chosen, last, targets = []) {
  const cells = [element("span", "axis"), ...Array.from(columns, (col) => element("span", "axis", col))];
  const claims = frame.sequences.map((sequences) => new Set(sequences.flat()));
  const legal = new Set(targets);
  replay.board.forEach((card, index) => {
    if (index % 10 === 0) cells.push(element("span", "axis", Math.floor(index / 10) + 1));

    const square = element("div", card === null ? "square corner" : "square");
    const chip = frame.chips[index];
    const description = [coordinate(index), card === null ? "Free corner" : cardName(card)];
    square.append(card === null ? "✦" : cardFace(card));
    square.classList.toggle("claimed-one", claims[0].has(index));
    square.classList.toggle("claimed-two", claims[1].has(index));
    square.classList.toggle("previous-play", index === last?.target);
    square.classList.toggle("recorded-target", index === chosen?.target);
    square.classList.toggle("legal-target", legal.has(index));
    if (chip !== null) {
      square.append(element("span", `chip ${seats[chip]}`));
      description.push(`${seatLabel(game, chip)} chip`);
    }
    if (legal.has(index)) description.push("Legal target");
    if (index === chosen?.target) description.push("Recorded target");
    square.title = description.join(" · ");
    cells.push(square);
  });
  byId("board").replaceChildren(...cells);
}

function renderHands(frame, chosen) {
  const showBoth = byId("reveal").checked;
  const visible = showBoth ? [0, 1] : [frame.active_player].filter((seat) => seat !== null);
  hands.replaceChildren(...visible.map((seat) => {
    const active = seat === frame.active_player;
    const row = element("div", "hand-row");
    row.append(element("p", "hand-label", seatLabel(game, seat) + (active ? " · Acting player" : "")));
    const cards = element("div", "hand");
    const recorded = active ? frame.hands[seat].findIndex(({ card }) => card === chosen?.card) : -1;
    for (const [index, info] of frame.hands[seat].entries()) {
      const card = active ? button("hand-card") : element("span", "hand-card");
      card.classList.toggle("dead", info.dead);
      card.classList.toggle("exchangeable", info.exchangeable);
      card.classList.toggle("recorded-card", index === recorded);
      card.append(cardFace(info.card));
      const label = [cardName(info.card)];
      if (info.dead) label.push("dead");
      if (info.exchangeable) label.push("exchange available");
      if (index === recorded) label.push("recorded choice");
      card.title = label.join(", ");
      if (active) {
        card.dataset.cardIndex = index;
        card.setAttribute("aria-label", card.title);
        card.setAttribute("aria-pressed", String(index === selectedCard));
      }
      cards.append(card);
    }
    if (!cards.childElementCount) cards.append(element("p", "hand-label", "No cards remaining."));
    row.append(cards);
    return row;
  }));
}

function showCardSelection() {
  const frame = replay.frames[frameIndex];
  const chosen = replay.actions[frameIndex];
  const last = replay.actions[frameIndex - 1];
  const info = frame.active_player === null ? undefined : frame.hands[frame.active_player][selectedCard];
  renderBoard(frame, chosen, last, info?.targets);

  for (const card of hands.querySelectorAll("button")) {
    card.setAttribute("aria-pressed", String(Number(card.dataset.cardIndex) === selectedCard));
  }
  const status = byId("card-status");
  if (!info) {
    status.textContent = chosen ? "Select a card to see its legal squares." : "";
  } else if (info.exchangeable) {
    status.textContent = `${info.card} is dead and can be exchanged.`;
  } else if (info.dead) {
    status.textContent = `${info.card} is dead. Exchange is unavailable.`;
  } else if (info.targets.length) {
    const removes = frame.chips[info.targets[0]] !== null;
    const count = info.targets.length;
    const noun = removes ? "removal" : "placement";
    status.textContent = `${info.card} · ${count} legal ${noun}${count === 1 ? "" : "s"}`;
  } else {
    status.textContent = `${info.card} has no legal targets.`;
  }
}

function renderTimeline() {
  const entries = replay.actions.map((action, index) => {
    const item = element("li");
    const step = button("");
    step.dataset.position = index;
    step.append(
      element("span", "action-number", index + 1),
      element("span", `player-dot ${seats[action.player]}`),
      actionText(action, replay.frames[index]),
    );
    item.append(step);
    return item;
  });
  const final = element("li");
  const end = button("", "Final position");
  end.dataset.position = replay.actions.length;
  final.append(end);
  actionsList.replaceChildren(...entries, final);
  actionsList.scrollTop = 0;
}

function setFrame(index) {
  frameIndex = Math.min(Math.max(0, index), replay.actions.length);
  selectedCard = null;
  const frame = replay.frames[frameIndex];
  const chosen = replay.actions[frameIndex];
  const position = chosen ? `Before action ${frameIndex + 1} of ${replay.actions.length}` : "Final position";
  byId("position-label").textContent = position;
  byId("decision").textContent = chosen
    ? `${seatLabel(game, chosen.player)} · ${actionText(chosen, frame)}`
    : `${replay.actions.length} actions recorded`;
  byId("previous").disabled = frameIndex === 0;
  byId("next").disabled = !chosen;
  scrubber.max = replay.actions.length;
  scrubber.value = frameIndex;
  scrubber.setAttribute("aria-valuetext", position);

  actionsList.querySelector("[aria-current]")?.removeAttribute("aria-current");
  const current = actionsList.children[frameIndex].firstElementChild;
  current.setAttribute("aria-current", "step");
  const viewport = actionsList.getBoundingClientRect();
  const bounds = current.getBoundingClientRect();
  if (bounds.top < viewport.top) actionsList.scrollTop += bounds.top - viewport.top;
  if (bounds.bottom > viewport.bottom) actionsList.scrollTop += bounds.bottom - viewport.bottom;

  renderPlayers(frame);
  renderHands(frame, chosen);
  showCardSelection();
  const params = new URLSearchParams({ game: game.game_id, action: chosen ? frameIndex + 1 : "end" });
  history.replaceState(null, "", `?${params}`);
}

async function openGame(row, position = 0) {
  pendingRequest?.abort();
  const request = new AbortController();
  pendingRequest = request;
  game = row;
  replay = undefined;
  markSelectedGame();
  replayContent.hidden = true;
  replayMessage.textContent = `Loading game #${row.game_id}…`;
  replayMessage.hidden = false;

  try {
    const loaded = await getJson(`/api/games/${row.game_id}`, request.signal);
    if (request.signal.aborted) return;
    replay = loaded;
    byId("game-title").textContent = `Game #${row.game_id}`;
    byId("game-details").textContent = `Pair #${row.pair_id} · Seed ${row.shuffle_seed} · ${row.actions} actions`;
    byId("paired").disabled = !pairedGame();
    renderGameResult();
    renderTimeline();
    replayContent.hidden = false;
    replayMessage.hidden = true;
    setFrame(position);
  } catch (error) {
    if (!request.signal.aborted) {
      replayContent.hidden = true;
      replayMessage.hidden = false;
      replayMessage.textContent = error.message;
    }
  }
}

function pairedGame() {
  return run.games.find((row) => row.pair_id === game.pair_id && row.policy_a_seat !== game.policy_a_seat);
}

async function init() {
  try {
    run = await getJson("/api/run");
  } catch (error) {
    byId("result-count").textContent = error.message;
    return;
  }
  byId("run-name").textContent = run.name;
  byId("matchup").textContent = `${policyLabel(0)} vs ${policyLabel(1)}`;
  filters.perspective.replaceChildren(...run.policies.map((_, index) => new Option(policyLabel(index), index)));
  byId("outcomes").replaceChildren(...Object.entries(outcomes).map(([key, [label]]) => {
    const option = button("", label);
    option.dataset.result = key;
    option.append(element("span", "count"));
    return option;
  }));
  renderCatalog();

  for (const filter of Object.values(filters)) filter.addEventListener("input", applyFilters);
  byId("outcomes").addEventListener("click", (event) => {
    const option = event.target.closest("button");
    if (!option) return;
    outcomeFilter = option.dataset.result;
    applyFilters();
  });
  byId("previous-page").addEventListener("click", () => {
    catalogPage--;
    renderCatalog();
  });
  byId("next-page").addEventListener("click", () => {
    catalogPage++;
    renderCatalog();
  });
  gameRows.addEventListener("click", (event) => {
    const row = event.target.closest("tr");
    if (row) openGame(run.games.find((game) => game.game_id === Number(row.dataset.game)));
  });
  byId("paired").addEventListener("click", () => {
    const paired = pairedGame();
    if (paired) openGame(paired);
  });
  byId("reveal").addEventListener("change", () => renderHands(replay.frames[frameIndex], replay.actions[frameIndex]));
  byId("previous").addEventListener("click", () => setFrame(frameIndex - 1));
  byId("next").addEventListener("click", () => setFrame(frameIndex + 1));
  scrubber.addEventListener("input", () => setFrame(Number(scrubber.value)));
  actionsList.addEventListener("click", (event) => {
    const step = event.target.closest("button");
    if (step) setFrame(Number(step.dataset.position));
  });
  hands.addEventListener("click", (event) => {
    const card = event.target.closest("button");
    if (!card) return;
    const index = Number(card.dataset.cardIndex);
    selectedCard = selectedCard === index ? null : index;
    showCardSelection();
  });

  const params = new URLSearchParams(location.search);
  if (params.has("game")) {
    const row = run.games.find((game) => game.game_id === Number(params.get("game")));
    const action = params.get("action");
    const number = Number(action);
    if (row) {
      const position = action === "end" ? row.actions : Number.isSafeInteger(number) && number > 0 ? number - 1 : 0;
      openGame(row, position);
    } else {
      replayMessage.textContent = "This game does not exist in the run.";
    }
  }
}

init();
