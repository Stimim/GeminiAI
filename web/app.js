'use strict';

const $ = id => document.getElementById(id);
const canvas = $('board');
const ctx = canvas.getContext('2d');
const tooltip = $('cellTooltip');
const tile = 40;
const colors = ['#2166ca', '#cf3546'];
const numberColors = ['', '#0000bb', '#087523', '#cb1724', '#000078', '#800000', '#007b7b', '#222222', '#747474'];
let game = null;
let gameId = null;
let hints = null;
let catalog = [];
let generation = 0;
let hintRequest = 0;
let busy = false;
let aiTimer = null;
let hovered = null;
let bombArmed = false;

function bombArea(cell) {
  const row = Math.floor(cell / game.w), col = cell % game.w;
  return {left: Math.max(0, col - 2), top: Math.max(0, row - 2),
    right: Math.min(game.w - 1, col + 2), bottom: Math.min(game.h - 1, row + 2)};
}
function bombLegal(cell) {
  const area = bombArea(cell);
  for (let r = area.top; r <= area.bottom; r++)
    for (let c = area.left; c <= area.right; c++)
      if (game.board[r * game.w + c] === -1) return true;
  return false;
}
function canBomb() {
  return game && game.winner === null && !busy && game.players[game.to_move] === 'human'
    && game.bombs[game.to_move] > 0;
}
function outlineArea(cell, color, preview = false) {
  const a = bombArea(cell);
  ctx.fillStyle = `${color}22`;
  ctx.fillRect(a.left * tile, a.top * tile, (a.right - a.left + 1) * tile, (a.bottom - a.top + 1) * tile);
  ctx.strokeStyle = color; ctx.lineWidth = 4;
  ctx.setLineDash(preview ? [9, 5] : []);
  ctx.strokeRect(a.left * tile + 2, a.top * tile + 2, (a.right - a.left + 1) * tile - 4, (a.bottom - a.top + 1) * tile - 4);
  ctx.setLineDash([]);
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : `Request failed (${response.status})`);
  return result;
}

function agentName(id) {
  return id === 'human' ? 'Human' : catalog.find(agent => agent.id === id)?.name || id;
}

function updateAbout(seat) {
  const id = $(`p${seat}`).value;
  $(`p${seat}About`).href = id === 'human' ? '/agents.html' : `/agents.html#${encodeURIComponent(id)}`;
  $(`p${seat}About`).title = id === 'human' ? 'Read about the available AI opponents.' : `Read about ${agentName(id)}.`;
}

function render() {
  if (!game) return;
  const finished = game.winner !== null;
  if (!canBomb()) bombArmed = false;
  $('bomb').style.visibility = canBomb() ? 'visible' : 'hidden';
  $('bomb').setAttribute('aria-pressed', String(bombArmed));
  $('bomb').textContent = bombArmed ? 'Armed — Esc to cancel' : 'Bomb (B)';
  $('bombStatus').textContent = bombArmed ? 'Click a centre to detonate a 5×5 bomb. Turn then passes.' : '';
  $('noBomb').style.visibility = game.bombs[game.to_move] === 0 ? 'visible' : 'hidden';
  $('bombStatus').style.visibility = bombArmed ? 'visible' : 'hidden';
  const target = Math.floor(game.mines / 2) + 1;
  const lastMove = game.last_move ? (game.last_move.kind === 'bomb'
    ? `Player ${game.last_move.player + 1} bomb · ${game.last_move.claimed.length} claimed, ${game.last_move.opened.length} revealed`
    : `Player ${game.last_move.player + 1} ${game.last_move.hit ? 'claimed a mine' : 'opened a safe cell'}`) : 'No moves yet';
  $('banner').textContent = finished ? 'Game over' :
    `Player ${game.to_move + 1}'s turn · ${agentName(game.players[game.to_move])}${busy ? ' · thinking…' : ''}`;
  $('banner').title = $('banner').textContent;
  $('lastMove').textContent = finished ? `Player ${game.winner + 1} wins!` : lastMove;
  $('lastMove').title = $('lastMove').textContent;
  $('banner').classList.toggle('game-over', finished);
  $('lastMove').classList.toggle('game-over', finished);
  game.scores.forEach((score, player) => {
    const panel = $('score').children[player];
    const name = panel.querySelector('.player-name');
    name.textContent = `Player ${player + 1} · ${agentName(game.players[player])}`;
    name.title = name.textContent;
    panel.querySelector('.player-points').textContent = score;
    panel.querySelector('.player-points').title = `Player ${player + 1} has claimed ${score} mines.`;
    const badge = panel.querySelector('.bomb-indicator');
    badge.textContent = game.bombs[player] ? 'Bomb available' : 'Bomb used';
    badge.classList.toggle('used', !game.bombs[player]);
    panel.classList.toggle('active', !finished && game.to_move === player);
  });
  $('target').textContent = `Target: ${target}`;
  $('target').title = `First to ${target} of the ${game.mines} mines wins.`;
  const value = hints?.value;
  $('value').textContent = `Value: ${Number.isFinite(value) ? (value >= 0 ? '+' : '') + value.toFixed(2) : '—'}`;
  const hasBombMoves = hints?.bomb_policy && Math.max(0, ...hints.bomb_policy) > 0;
  $('policyKind').options[1].disabled = !hasBombMoves;
  if (!hasBombMoves && $('policyKind').value === 'bomb') $('policyKind').value = 'open';
  const bombPolicy = $('showPolicy').checked && $('policyKind').value === 'bomb' && hasBombMoves ? hints?.bomb_policy : null;
  $('policyKind').style.visibility = !$('hintOn').checked || !$('showPolicy').checked || !hints?.bomb_policy ? 'hidden' : 'visible';
  const distribution = $('hintOn').checked ? ($('showPolicy').checked ? (bombPolicy || hints?.policy) : hints?.mine) : null;
  $('hintStatus').textContent = $('hintOn').checked && hints
    ? ($('showPolicy').checked ? (distribution ? `${bombPolicy ? 'Bomb-centre' : 'Open'} policy %${!hasBombMoves && hints?.bomb_policy ? ' · No bomb available' : ''}` : 'Policy unavailable for this agent') : (hints.mine ? 'Mine %' : 'Mine probabilities unavailable for this agent')) : '';
  canvas.width = game.w * tile;
  canvas.height = game.h * tile;
  const maxProbability = distribution ? Math.max(0.001, ...distribution.filter(Number.isFinite)) : 1;
  game.board.forEach((cell, index) => {
    const x = index % game.w * tile;
    const y = Math.floor(index / game.w) * tile;
    if (cell === -1) {
      ctx.fillStyle = '#9ba9b9';
      ctx.fillRect(x, y, tile, tile);
      ctx.fillStyle = '#d8e0e9';
      ctx.beginPath(); ctx.moveTo(x, y + tile); ctx.lineTo(x, y); ctx.lineTo(x + tile, y); ctx.lineTo(x + tile - 4, y + 4); ctx.lineTo(x + 4, y + 4); ctx.lineTo(x + 4, y + tile - 4); ctx.fill();
      ctx.fillStyle = '#5b6a7c';
      ctx.beginPath(); ctx.moveTo(x + tile, y); ctx.lineTo(x + tile, y + tile); ctx.lineTo(x, y + tile); ctx.lineTo(x + 4, y + tile - 4); ctx.lineTo(x + tile - 4, y + tile - 4); ctx.lineTo(x + tile - 4, y + 4); ctx.fill();
    } else {
      ctx.fillStyle = '#e9edf1'; ctx.fillRect(x, y, tile, tile);
      ctx.strokeStyle = '#c0cad4'; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, tile - 1, tile - 1);
      ctx.fillStyle = '#d6dee6'; ctx.fillRect(x + 1, y + 1, tile - 2, 1);
    }
    if ((cell === -1 || bombPolicy) && distribution && Number.isFinite(distribution[index])) {
      const probability = Math.max(0, Math.min(1, distribution[index]));
      const intensity = $('showPolicy').checked ? probability / maxProbability : probability;
      ctx.fillStyle = bombPolicy ? `hsl(285 72% ${92 - intensity * 40}%)` : `hsl(${205 - intensity * 185} 78% ${89 - intensity * 26}%)`;
      ctx.fillRect(x + 4, y + 4, tile - 8, tile - 8);
      if (cell === -1 && probability >= 0.005) {
        ctx.fillStyle = '#142231'; ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(`${(probability * 100).toFixed(probability < .1 ? 1 : 0)}%`, x + tile / 2, y + tile / 2);
      }
    }
    if (cell > 0 && cell <= 8) {
      ctx.fillStyle = numberColors[cell]; ctx.font = 'bold 25px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(cell, x + tile / 2, y + tile / 2 + 1);
    } else if (cell >= 9) {
      ctx.strokeStyle = '#29323e'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x + 17, y + 9); ctx.lineTo(x + 17, y + 30); ctx.moveTo(x + 11, y + 31); ctx.lineTo(x + 28, y + 31); ctx.stroke();
      ctx.fillStyle = colors[cell - 9]; ctx.beginPath(); ctx.moveTo(x + 18, y + 9); ctx.lineTo(x + 31, y + 14); ctx.lineTo(x + 18, y + 21); ctx.fill();
    } else if (finished && cell === -1 && game.mine_layout?.[index]) {
      ctx.fillStyle = '#29323e'; ctx.beginPath(); ctx.arc(x + 20, y + 20, 8, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#29323e'; ctx.lineWidth = 2;
      for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 4) {
        ctx.beginPath(); ctx.moveTo(x + 20 + 6 * Math.cos(angle), y + 20 + 6 * Math.sin(angle)); ctx.lineTo(x + 20 + 12 * Math.cos(angle), y + 20 + 12 * Math.sin(angle)); ctx.stroke();
      }
    }
    if (game.last_move?.cell === index) {
      ctx.strokeStyle = '#ffe245'; ctx.lineWidth = 3; ctx.strokeRect(x + 2, y + 2, tile - 4, tile - 4);
    }
  });
  if (game.last_move?.kind === 'bomb') outlineArea(game.last_move.cell, colors[game.last_move.player]);
  if (bombArmed && hovered) outlineArea(hovered.cell, bombLegal(hovered.cell) ? '#a85a00' : '#b00020', true);
  canvas.style.cursor = !finished && !busy && game.players[game.to_move] === 'human' ? 'pointer' : 'default';
  if (bombArmed) canvas.style.cursor = hovered && !bombLegal(hovered.cell) ? 'not-allowed' : 'crosshair';
  if (hovered) showCellTooltip(hovered);
}

async function refreshHints(token) {
  const request = ++hintRequest;
  hints = null;
  render();
  if (!$('hintOn').checked || !game || game.winner !== null) return;
  $('hintStatus').textContent = 'Calculating hints…';
  try {
    const result = await api(`/api/games/${gameId}/hint?agent=${encodeURIComponent($('hintAgent').value)}`);
    if (token !== generation || request !== hintRequest) return;
    hints = result;
    render();
  } catch (error) {
    if (token === generation && request === hintRequest) {
      $('error').textContent = error.message;
      $('hintStatus').textContent = 'Hints unavailable';
    }
  }
}

function scheduleAI(token) {
  clearTimeout(aiTimer);
  if (!game || game.winner !== null || game.players[game.to_move] === 'human') return;
  aiTimer = setTimeout(() => makeMove('ai', {}, token), 300);
}

async function makeMove(action, body, token = generation) {
  if (busy || token !== generation || !game || game.winner !== null) return;
  busy = true;
  $('error').textContent = '';
  ++hintRequest;
  hints = null;
  render();
  try {
    const result = await api(`/api/games/${gameId}/${action}`, body);
    if (token !== generation) return;
    game = result; // Refreshes omit the id; gameId deliberately remains unchanged.
    bombArmed = false;
    busy = false;
    render();
    await refreshHints(token);
    if (token === generation) scheduleAI(token);
  } catch (error) {
    if (token === generation) { busy = false; $('error').textContent = error.message; render(); }
  }
}

async function newGame(event) {
  event?.preventDefault();
  const token = ++generation;
  ++hintRequest;
  clearTimeout(aiTimer);
  hints = null;
  bombArmed = false;
  busy = true;
  $('new').disabled = true;
  $('error').textContent = '';
  tooltip.hidden = true;
  hovered = null;
  try {
    const result = await api('/api/games', {
      players: [$('p0').value, $('p1').value], sims: Number($('sims').value),
      seed: $('seed').value === '' ? null : Number($('seed').value)
    });
    if (token !== generation) return;
    gameId = result.id;
    game = result;
    filterAgents();
    busy = false;
    render();
    await refreshHints(token);
    if (token === generation) scheduleAI(token);
  } catch (error) {
    if (token === generation) { busy = false; $('error').textContent = error.message; render(); }
  } finally {
    if (token === generation) $('new').disabled = false;
  }
}

function cellAt(event) {
  if (!game) return null;
  const rect = canvas.getBoundingClientRect();
  const col = Math.floor((event.clientX - rect.left) / rect.width * game.w);
  const row = Math.floor((event.clientY - rect.top) / rect.height * game.h);
  return col >= 0 && col < game.w && row >= 0 && row < game.h ? row * game.w + col : null;
}

function showCellTooltip(position) {
  if (!game) return;
  const cell = game.board[position.cell];
  const state = cell === -1 ? (game.mine_layout?.[position.cell] ? 'Unclaimed mine' : 'Unknown')
    : cell >= 9 ? `Mine claimed by Player ${cell - 8}` : cell === 0 ? 'Opened · empty (0 nearby mines)' : `Opened · ${cell} nearby mine${cell === 1 ? '' : 's'}`;
  const lines = [`Row ${Math.floor(position.cell / game.w) + 1}, column ${position.cell % game.w + 1} · ${state}`];
  if ($('hintOn').checked && Number.isFinite(hints?.mine?.[position.cell])) lines.push(`Mine probability: ${(hints.mine[position.cell] * 100).toFixed(1)}%`);
  if ($('hintOn').checked && Number.isFinite(hints?.policy?.[position.cell])) lines.push(`Policy: ${(hints.policy[position.cell] * 100).toFixed(2)}%`);
  if ($('hintOn').checked && Number.isFinite(hints?.q?.[position.cell])) lines.push(`Search Q: ${hints.q[position.cell].toFixed(2)}`);
  if ($('hintOn').checked && Number.isFinite(hints?.bomb_policy?.[position.cell])) lines.push(`Bomb-centre policy: ${(hints.bomb_policy[position.cell] * 100).toFixed(2)}%`);
  if ($('hintOn').checked && Number.isFinite(hints?.bomb_q?.[position.cell])) lines.push(`Bomb search Q: ${hints.bomb_q[position.cell].toFixed(2)}`);
  if (bombArmed) lines.push(bombLegal(position.cell) ? 'Click to detonate · 5×5 area · turn passes' : 'Illegal bomb target: area fully opened');
  tooltip.textContent = lines.join('\n');
  tooltip.hidden = false;
  tooltip.style.left = '0px';
  tooltip.style.top = `${position.y + 16}px`;
  const width = canvas.getBoundingClientRect().width;
  tooltip.style.left = `${Math.max(0, Math.min(position.x + 12, width - tooltip.offsetWidth))}px`;
  if (position.y + 16 + tooltip.offsetHeight > canvas.getBoundingClientRect().height) tooltip.style.top = `${Math.max(0, position.y - tooltip.offsetHeight - 10)}px`;
}

canvas.addEventListener('mousemove', event => {
  const cell = cellAt(event);
  if (cell === null) return;
  const rect = canvas.getBoundingClientRect();
  hovered = {cell, x: event.clientX - rect.left, y: event.clientY - rect.top};
  render();
});
canvas.addEventListener('mouseleave', () => { hovered = null; tooltip.hidden = true; render(); });
canvas.addEventListener('click', event => {
  const cell = cellAt(event);
  if (cell === null || busy || game.winner !== null || game.players[game.to_move] !== 'human') return;
  if (bombArmed) {
    if (canBomb() && bombLegal(cell)) makeMove('move', {cell, bomb: true});
  } else if (game.board[cell] === -1) makeMove('move', {cell, bomb: false});
});
$('newGameForm').addEventListener('submit', newGame);
$('hintOn').addEventListener('change', () => refreshHints(generation));
$('hintAgent').addEventListener('change', () => refreshHints(generation));
$('showPolicy').addEventListener('change', render);
$('policyKind').addEventListener('change', render);
$('bomb').addEventListener('click', () => { if (canBomb()) { bombArmed = !bombArmed; render(); } });
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input, select, textarea, [contenteditable="true"]')) return;
  if (event.key === 'Escape') { bombArmed = false; render(); }
  else if (event.key.toLowerCase() === 'b' && !event.repeat && canBomb()) {
    event.preventDefault(); bombArmed = !bombArmed; render();
  }
});

function filterAgents(initial = false) {
  const available = catalog;
  for (const id of ['p0', 'p1', 'hintAgent']) {
    const select = $(id), previous = select.value;
    select.replaceChildren();
    if (id !== 'hintAgent') select.add(new Option('Human', 'human'));
    for (const agent of available) {
      const option = new Option(agent.name, agent.id);
      option.title = agent.summary;
      select.add(option);
    }
    if ([...select.options].some(option => option.value === previous)) select.value = previous;
  }
  if (initial) $('p1').value = available.find(agent => agent.request_sims)?.id || available.find(agent => agent.id === 'solver')?.id || 'human';
  if (initial || !$('hintAgent').value) $('hintAgent').value = available.find(agent => agent.id === 'solver')?.id || available[0]?.id || '';
  $('hintOn').disabled = available.length === 0;
  [0, 1].forEach(updateAbout);
}
[0, 1].forEach(seat => $(`p${seat}`).addEventListener('change', () => updateAbout(seat)));

async function initialize() {
  try {
    const result = await api('/api/agents');
    catalog = result.agents.filter(agent => agent.available !== false);
    filterAgents(true);
    await newGame();
  } catch (error) { $('error').textContent = error.message; }
}
initialize();
