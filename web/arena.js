'use strict';

const $ = id => document.getElementById(id);
const agents = new Map();
let categories = [];
let matchId = null;
let pollTimer = null;
let pollGeneration = 0;
let activeMatch = false;
let catalogReady = false;
let submitting = false;
let lastResultCount = -1;
let ratingsRequest = 0;

function selectedContender(id) {
  return AgentCatalog.selected($(id), $(`${id}-generation`), categories);
}
function updateGenerationPicker(id) {
  const category = categories.find(item => item.id === $(id).value);
  AgentCatalog.fillGenerations($(`${id}-generation`), category);
  $(`${id}-generation`).closest('.generation-picker').hidden = !AgentCatalog.neural(category);
}
function filterContenders() {
  const available = categories.filter(category => category.available);
  for (const id of ['agent-a', 'agent-b']) {
    $(id).replaceChildren(...categories.map(category => {
      const option = node('option', '', category.name);
      option.value = category.id; option.disabled = !category.available; return option;
    }));
    $(id).value = available[0]?.id || '';
  }
  $('agent-b').value = available[1]?.id || available[0]?.id || '';
  ['agent-a', 'agent-b'].forEach(updateGenerationPicker);
  catalogReady = [...agents.values()].filter(agent => agent.available).length >= 2;
  errorAt('setup-error', catalogReady ? '' : 'At least two AI generations are required.');
  validateContenders();
  updateControls();
}

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}
function errorAt(id, message) { $(id).textContent = message; $(id).hidden = !message; }
function name(id) { return agents.get(id)?.name || id || 'Unknown agent'; }
async function api(path, options = {}) {
  const response = await fetch(path, {headers: {Accept: 'application/json', 'Content-Type': 'application/json'}, ...options});
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const detail = data.detail || data.error || response.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}
function updateControls() {
  for (const id of ['agent-a', 'agent-b', 'game-count']) $(id).disabled = !catalogReady || activeMatch || submitting;
  for (const id of ['agent-a', 'agent-b']) $(`${id}-generation`).disabled = !catalogReady || activeMatch || submitting || !AgentCatalog.neural(categories.find(category => category.id === $(id).value));
  $('start-match').disabled = !catalogReady || activeMatch || submitting;
  $('start-match').textContent = submitting ? 'Starting…' : activeMatch ? 'Match running…' : 'Start match';
}
async function loadAgents() {
  try {
    const data = await api('/api/agents');
    const list = Array.isArray(data) ? data : data.agents;
    if (!Array.isArray(list)) throw new Error('The server did not return an agent list.');
    categories = list;
    list.forEach(category => category.generations.forEach(generation => agents.set(generation.id, generation)));
    filterContenders();
    const requested = new URLSearchParams(location.search).get('a');
    if (requested) AgentCatalog.setSelection($('agent-a'), $('agent-a-generation'), categories, requested);
    ['agent-a', 'agent-b'].forEach(id => {
      $(`${id}-generation`).closest('.generation-picker').hidden = !AgentCatalog.neural(categories.find(category => category.id === $(id).value));
    });
    if (selectedContender('agent-b') === selectedContender('agent-a')) {
      const alternative = [...agents.values()].find(agent => agent.available && agent.id !== selectedContender('agent-a'));
      if (alternative) AgentCatalog.setSelection($('agent-b'), $('agent-b-generation'), categories, alternative.id);
    }
    validateContenders();
    if (matchId) resumeMatch(matchId);
  } catch (error) { errorAt('setup-error', `Could not load contenders: ${error.message}`); }
}

function renderBoard(game) {
  const board = $('arena-board');
  if (!game) {
    board.hidden = true;
    $('scoreboard').hidden = true;
    $('board-meta').textContent = '';
    $('board-status').textContent = 'Waiting for the first game…';
    return;
  }
  const target = Math.floor(game.mines / 2) + 1;
  $('board-meta').textContent = `${game.h} × ${game.w} · ${game.mines} mines · ${game.moves} moves`;
  const playerNames = game.players.map(name);
  $('board-status').textContent = game.winner !== null && game.winner !== undefined
    ? `${playerNames[game.winner]} wins this game.`
    : `${playerNames[game.to_move]} to move · first to ${target} mines wins.`;
  $('scoreboard').hidden = false;
  $('scoreboard').replaceChildren(...game.scores.map((score, seat) => {
    const scorecard = node('div', `scorecard player-${seat === 0 ? 'one' : 'two'}${game.to_move === seat && game.winner === null ? ' current-player' : ''}`);
    scorecard.append(node('span', '', `Player ${seat + 1} · ${playerNames[seat]}`), node('strong', '', score));
    const indicator = node('small', 'bomb-indicator', game.bombs[seat] ? 'Bomb available' : 'Bomb used');
    indicator.title = 'Each player has one 5×5 bomb. Detonating claims unopened mines and passes the turn.';
    scorecard.append(indicator);
    return scorecard;
  }));
  board.style.setProperty('--board-cols', game.w);
  board.setAttribute('aria-rowcount', game.h);
  board.setAttribute('aria-colcount', game.w);
  const lastCell = game.last_move?.cell;
  const rows = [];
  for (let r = 0; r < game.h; r++) {
    const row = node('div', 'board-row');
    row.setAttribute('role', 'row');
    for (let c = 0; c < game.w; c++) {
      const i = r * game.w + c;
      const value = game.board[i];
      const claimed = value === 9 || value === 10;
      const cell = node('span', `board-cell ${value === -1 ? 'covered' : claimed ? `claimed player-${value === 9 ? 'one' : 'two'}` : `open number-${value}`}${lastCell === i ? ' last-move' : ''}`, claimed ? '⚑' : value > 0 ? value : '');
      if (game.last_move?.kind === 'bomb' && Math.abs(r - Math.floor(lastCell / game.w)) <= 2 && Math.abs(c - lastCell % game.w) <= 2) cell.classList.add('last-bomb');
      cell.setAttribute('role', 'cell');
      cell.setAttribute('aria-label', `Row ${r + 1}, column ${c + 1}: ${value === -1 ? 'unopened' : claimed ? `mine claimed by player ${value - 8}` : value === 0 ? 'empty' : `${value} adjacent mines`}${lastCell === i ? ', last move' : ''}`);
      row.append(cell);
    }
    rows.push(row);
  }
  board.replaceChildren(...rows);
  board.hidden = false;
}

function renderResults(match) {
  const results = match.results || [];
  // A match appends immutable records; avoid rebuilding the results table on every move.
  if (lastResultCount === results.length) return;
  lastResultCount = results.length;
  const winsA = results.filter(result => result.winner === match.a).length;
  const winsB = results.filter(result => result.winner === match.b).length;
  const summary = $('results-summary');
  if (!results.length) summary.textContent = 'No completed games yet.';
  else summary.textContent = `${name(match.a)} ${winsA} – ${winsB} ${name(match.b)}`;
  const table = node('table');
  table.append(node('caption', 'sr-only', 'Completed games in this match'));
  const head = node('thead');
  const heading = node('tr');
  for (const text of ['Game', 'Winner', 'Score (P1–P2)']) {
    const th = node('th', '', text); th.scope = 'col'; heading.append(th);
  }
  head.append(heading);
  const body = node('tbody');
  for (const result of results) {
    const row = node('tr');
    row.append(node('td', '', result.game), node('td', '', name(result.winner)), node('td', 'numeric', result.scores.join(' – ')));
    body.append(row);
  }
  table.append(head, body);
  $('results-table').replaceChildren(...(results.length ? [table] : []));
}
function renderMatch(match) {
  $('match-section').hidden = false;
  $('match-title').textContent = `${name(match.a)} vs ${name(match.b)}`;
  if (agents.has(match.a) && agents.has(match.b)) {
    AgentCatalog.setSelection($('agent-a'), $('agent-a-generation'), categories, match.a);
    AgentCatalog.setSelection($('agent-b'), $('agent-b-generation'), categories, match.b);
    for (const id of ['agent-a', 'agent-b']) $(`${id}-generation`).closest('.generation-picker').hidden = !AgentCatalog.neural(categories.find(category => category.id === $(id).value));
    $('game-count').value = match.games;
  }
  $('match-status').textContent = match.status;
  $('match-status').className = `badge status-${match.status}`;
  $('match-progress').max = match.games;
  $('match-progress').value = match.completed;
  $('match-progress-text').textContent = `${match.completed} of ${match.games} games completed${match.status === 'completed' ? ' · Match finished.' : match.status === 'failed' ? ' · Match stopped.' : ' · Live updates every second.'}`;
  errorAt('match-error', match.error ? `Match failed: ${typeof match.error === 'string' ? match.error : JSON.stringify(match.error)}` : '');
  activeMatch = match.status === 'running';
  renderBoard(match.current);
  renderResults(match);
  updateControls();
}
async function pollMatch(generation) {
  try {
    const match = await api(`/api/arena/matches/${encodeURIComponent(matchId)}`);
    if (generation !== pollGeneration) return;
    $('retry-match').hidden = true;
    renderMatch(match);
    if (activeMatch) pollTimer = setTimeout(() => pollMatch(generation), 1000);
    else loadRatings();
  } catch (error) {
    if (generation !== pollGeneration) return;
    $('match-section').hidden = false;
    errorAt('match-error', `Live updates paused: ${error.message} The server may still be running the match. Resume updates to reconnect.`);
    $('retry-match').hidden = false;
  }
}
function resumeMatch(id) {
  clearTimeout(pollTimer);
  matchId = id;
  lastResultCount = -1;
  activeMatch = true;
  updateControls();
  pollMatch(++pollGeneration);
}
function validateContenders() {
  $('agent-b-generation').setCustomValidity(selectedContender('agent-a') === selectedContender('agent-b') ? 'Choose two different generations.' : '');
  $('agent-b').setCustomValidity(selectedContender('agent-a') === selectedContender('agent-b') ? 'Choose two different generations.' : '');
}
for (const id of ['agent-a', 'agent-b']) {
  $(id).addEventListener('change', () => { updateGenerationPicker(id); validateContenders(); updateControls(); });
  $(`${id}-generation`).addEventListener('change', validateContenders);
}
$('match-form').addEventListener('submit', async event => {
  event.preventDefault();
  validateContenders();
  if (activeMatch || submitting || !$('match-form').reportValidity()) return;
  submitting = true;
  updateControls();
  errorAt('setup-error', '');
  try {
    const match = await api('/api/arena/matches', {method: 'POST', body: JSON.stringify({a: selectedContender('agent-a'), b: selectedContender('agent-b'), games: Number($('game-count').value)})});
    matchId = match.id;
    lastResultCount = -1;
    history.replaceState(null, '', `${location.pathname}${location.search}#${encodeURIComponent(matchId)}`);
    renderMatch(match);
    $('match-section').scrollIntoView({block: 'start'});
    if (activeMatch) pollMatch(++pollGeneration);
    else loadRatings();
  } catch (error) { errorAt('setup-error', `Could not start the match: ${error.message}`); }
  finally { submitting = false; updateControls(); }
});
$('retry-match').addEventListener('click', () => resumeMatch(matchId));

async function loadRatings() {
  const request = ++ratingsRequest;
  $('refresh-ratings').disabled = true;
  errorAt('ratings-error', '');
  $('ratings-status').textContent = 'Loading ratings…';
  try {
    const data = await api('/api/ratings');
    if (request !== ratingsRequest) return;
    const ratings = new Map(data.ratings.map(rating => [rating.id, rating]));
    const rows = [];
    const ranked = data.ratings.filter(rating => rating.games > 0).sort((a, b) => b.rating - a.rating || a.id.localeCompare(b.id));
    for (const category of categories) {
      const group = node('tr', 'category-header');
      const heading = node('th');
      heading.colSpan = 4; heading.scope = 'rowgroup';
      const categoryLink = node('a', 'text-link', category.name);
      categoryLink.href = `/agents.html#${encodeURIComponent(category.id)}`;
      heading.append(categoryLink); group.append(heading); rows.push(group);
      for (const generation of AgentCatalog.generations(category)) {
        const rating = ratings.get(generation.id) || generation;
        const row = node('tr', 'generation-row');
        const agent = node('td');
        const link = node('a', 'text-link', generation.label || generation.name);
        link.href = `/agents.html#${encodeURIComponent(generation.id)}`;
        agent.append(link);
        if (generation.id === data.anchor) agent.append(node('span', 'badge anchor-badge', 'Anchor'));
        if (AgentCatalog.neural(category) && generation.id === category.default_generation) agent.append(node('span', 'badge anchor-badge', 'Latest'));
        if (!generation.available) agent.append(node('span', 'badge unavailable', 'Unavailable'));
        const value = node('td', 'rating-value', AgentCatalog.elo(rating));
        if (Number.isFinite(rating.rating) && Number.isFinite(rating.uncertainty)) value.title = `95% interval: ${Math.round(rating.rating - rating.uncertainty)} to ${Math.round(rating.rating + rating.uncertainty)} Elo`;
        const rank = ranked.findIndex(item => item.id === generation.id);
        row.append(node('td', 'muted', rank < 0 ? '—' : rank + 1), agent, value, node('td', 'numeric', rating.games ?? 0));
        rows.push(row);
      }
    }
    $('ratings-body').replaceChildren(...rows);
    $('ratings-table').hidden = !rows.length;
    $('ratings-status').textContent = rows.length ? 'Grouped by category; newest generations first. Ratings use all recorded arena games.' : 'No generations published yet.';
  } catch (error) {
    if (request !== ratingsRequest) return;
    $('ratings-status').textContent = 'Ratings could not be refreshed. Any ratings shown below are from the previous update.';
    errorAt('ratings-error', error.message);
  } finally { if (request === ratingsRequest) $('refresh-ratings').disabled = false; }
}
$('refresh-ratings').addEventListener('click', loadRatings);
window.addEventListener('hashchange', () => {
  if (location.hash.length > 1) {
    try { resumeMatch(decodeURIComponent(location.hash.slice(1))); }
    catch { errorAt('setup-error', 'The match link contains an invalid ID.'); }
  }
});
loadAgents().then(loadRatings);
if (location.hash.length > 1) {
  try { resumeMatch(decodeURIComponent(location.hash.slice(1))); }
  catch { errorAt('setup-error', 'The match link contains an invalid ID.'); }
}
