'use strict';

// Tensor dimensions omit the batch axis. Nodes mirror Net.forward in model.py.
function architectureSection(agent) {
  const section = element('section', 'detail-section architecture');
  section.append(element('h3', '', 'Network architecture'));
  if (!agent.shape || !agent.board || !agent.input_planes) {
    section.append(element('p', 'muted', 'Architecture unavailable: this checkpoint is not installed.'));
    if (agent.kind === 'mcts') section.append(searchSection(agent));
    return section;
  }
  const {channels: c, blocks: n, policy_beta: initialBeta} = agent.shape;
  const {h, w} = agent.board;
  const planes = agent.input_planes;
  const tensor = `${c} × ${h} × ${w}`;
  const cells = `${h} × ${w}`;
  const beta = agent.policy_beta ?? initialBeta;
  section.append(element('p', 'arch-summary', `${c} channels · ${n} residual blocks · ${Number(agent.parameters).toLocaleString()} learned parameters`));
  section.append(element('p', 'muted small', 'Checkpoint-derived dimensions; batch axis omitted. The same convolutional weights support other board sizes.'));
  const grid = element('div', 'arch-grid');
  const ns = 'http://www.w3.org/2000/svg';
  function svgPanel(label) {
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 360 640');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', label);
    svg.classList.add('arch-svg');
    const title = document.createElementNS(ns, 'title');
    title.textContent = label;
    svg.append(title);
    return svg;
  }
  function draw(svg, tag, attrs, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  }
  function label(svg, x, y, text, cls = '') {
    return draw(svg, 'text', {x, y, 'text-anchor': 'middle', class: cls}, text);
  }
  function box(svg, x, y, width, height, title, lines, cls = '') {
    draw(svg, 'rect', {x, y, width, height, rx: 10, class: `arch-node ${cls}`});
    label(svg, x + width / 2, y + 25, title, 'arch-node-title');
    lines.forEach((line, i) => label(svg, x + width / 2, y + 49 + i * 19, line));
  }
  function arrow(svg, points, cls = '') {
    draw(svg, 'polyline', {points, class: `arch-edge ${cls}`});
    const pairs = points.split(' ').map(pair => pair.split(',').map(Number));
    const [x, y] = pairs[pairs.length - 1];
    const [px, py] = pairs[pairs.length - 2];
    const angle = Math.atan2(y - py, x - px);
    const wing = side => `${x - 7 * Math.cos(angle) + side * 3.5 * Math.sin(angle)},${y - 7 * Math.sin(angle) - side * 3.5 * Math.cos(angle)}`;
    draw(svg, 'polygon', {points: `${x},${y} ${wing(1)} ${wing(-1)}`, class: `arch-arrow ${cls}`});
  }
  const trunk = svgPanel(`Input ${planes} planes, stem ${planes} to ${c} channels, and ${n} residual blocks.`);
  label(trunk, 180, 22, 'PUBLIC BOARD → SHARED FEATURES', 'arch-kicker');
  box(trunk, 32, 42, 296, 78, 'Input encoding', [`${planes} planes × ${cells}`, 'Current player’s perspective']);
  arrow(trunk, '180,120 180,157');
  label(trunk, 251, 143, `${planes} × ${cells}`, 'arch-shape');
  box(trunk, 32, 157, 296, 96, 'Stem', [`3×3 conv: ${planes} → ${c}`, 'BatchNorm → ReLU', `Output: ${tensor}`]);
  arrow(trunk, '180,253 180,288');
  label(trunk, 250, 276, tensor, 'arch-shape');
  draw(trunk, 'rect', {x: 20, y: 288, width: 320, height: 273, rx: 12, class: 'arch-tower'});
  label(trunk, 180, 314, `RESIDUAL TOWER × ${n}`, 'arch-kicker');
  const shown = n <= 3 ? Array.from({length: n}, (_, i) => i + 1) : [1, 2, '…', n];
  const slot = 198 / Math.max(shown.length, 1);
  shown.forEach((block, i) => {
    const y = 330 + i * slot;
    if (block === '…') {
      label(trunk, 180, y + 22, `… ${n - 3} more block${n - 3 === 1 ? '' : 's'} …`, 'arch-shape');
    } else {
      draw(trunk, 'rect', {x: 49, y, width: 262, height: slot - 9, rx: 7, class: 'arch-node'});
      label(trunk, 180, y + (slot - 9) / 2 + 5, `Block ${block} · ${tensor}`, 'arch-node-title');
    }
    if (i < shown.length - 1) arrow(trunk, `180,${y + slot - 9} 180,${y + slot}`);
  });
  label(trunk, 180, 546, 'Each block preserves the tensor shape', 'arch-shape');
  arrow(trunk, '180,561 180,600');
  label(trunk, 180, 623, `${tensor} → all four heads`, 'arch-node-title');
  grid.append(trunk);

  const heads = svgPanel('Mine probability, open policy with beta times mine-logit connection, and pooled value. The bomb policy head is described below.');
  label(heads, 180, 22, `SHARED FEATURES · ${tensor}`, 'arch-kicker');
  draw(heads, 'polyline', {points: '18,34 18,500', class: 'arch-edge'});
  arrow(heads, '18,82 40,82');
  arrow(heads, '18,290 40,290');
  arrow(heads, '18,500 40,500');
  box(heads, 40, 42, 290, 124, 'Mine head', [`1×1 conv: ${c} → 32 → ReLU`, '1×1 conv: 32 → 1', `Mine logit · ${cells}`, 'sigmoid → P(mine) per cell'], 'arch-mine');
  arrow(heads, '330,112 347,112 347,213 185,213 185,239', 'arch-beta');
  label(heads, 183, 192, `β × mine logit · β = ${Number(beta).toPrecision(4)}`, 'arch-beta-label');
  box(heads, 40, 239, 290, 163, 'Open policy head', [`Residual: 1×1 conv ${c} → 32`, 'ReLU → 1×1 conv 32 → 1', 'Final residual layer starts at zero', '+ β × mine logit', 'Join bomb logits → masked softmax', `${h * w} open-cell probabilities`], 'arch-policy');
  label(heads, 184, 428, `Per-cell logits: 1 × ${cells}`, 'arch-shape');
  box(heads, 40, 456, 290, 151, 'Value head', ['Board-masked mean + spatial max', `Concatenate → ${2 * c} features (2C)`, `Linear ${2 * c} → 64 → ReLU`, 'Linear 64 → 1 → tanh', 'Outcome for player to move: [−1, 1]'], 'arch-value');
  label(heads, 180, 632, 'Arrows carry features unless labelled', 'arch-shape');
  grid.append(heads);
  section.append(grid);
  const bomb = element('section', 'arch-inputs');
  bomb.append(element('h3', '', 'Bomb policy head'),
    element('p', '', `Shared features → 1×1 conv ${c} → 32 → ReLU → 1×1 conv 32 → 1 residual, initialized to zero. Add a learned scale times (expected unopened mines in a clipped 5×5 window − threshold). Open and bomb logits share one masked softmax over ${2 * h * w} actions. A bomb centre is legal when the player has a bomb and its window contains an unopened cell, even if the centre is open.`));
  section.append(bomb);
  const residual = element('p', 'arch-caption');
  residual.append(element('strong', '', 'Inside each residual block: '), document.createTextNode(`3×3 conv ${c}→${c} → BN → ReLU → 3×3 conv ${c}→${c} → BN → add skip input → ReLU.`));
  section.append(residual);
  const input = element('details', 'arch-inputs');
  input.append(element('summary', '', `What are the ${planes} input planes?`));
  input.append(element('p', '', '0 unknown; 1 claimed by me; 2 claimed by opponent; 3–11 opened counts 0–8 (one-hot); 12 residual mines around opened cells (clue − claimed neighbours) / 8; 13 unknown-neighbour count / 8; 14 my score / winning score; 15 opponent score / winning score; 16 remaining mines / unknown cells; 17 board mask (ones on the board). “Me” is the player to move.'));
  input.append(element('p', '', '18 my remaining bomb; 19 opponent’s remaining bomb. Bomb availability is encoded across the board.'));
  section.append(input);
  section.append(element('p', 'arch-caption', 'β is a learned scalar; the arrow shows its checkpoint value (initial value comes from net_cfg.policy_beta). Mean pooling uses the board mask; max pooling is over the full spatial tensor. Mine sigmoid and policy softmax are applied by the evaluator.'));
  if (agent.kind === 'mcts') section.append(searchSection(agent));
  return section;
}

function searchSection(agent) {
  const section = element('div', 'arch-search');
  section.append(element('h3', '', 'How the network guides search'));
  const flow = element('div', 'search-flow');
  ['Sample a clue-consistent board', 'Select moves with PUCT', 'Evaluate a leaf with the net', 'Policy → priors · value → backup', 'Choose the most-visited move'].forEach((text, i) => {
    if (i) flow.append(element('span', 'search-flow-arrow', '↓'));
    flow.append(element('span', 'search-flow-node', text));
  });
  section.append(flow);
  section.append(element('p', 'arch-caption', 'Repeat simulations from public information, never the real hidden mine layout. Terminal states back up the actual outcome; other leaves use the net’s value. Backups preserve the value perspective when a mine keeps the turn and switch perspective when the player changes.'));
  const glossary = element('dl', 'search-glossary');
  const settings = agent.settings || {};
  const entries = [
    [`sims · ${settings.sims ?? 'not provided'}`, 'Simulation budget per move. The initial root network evaluation also counts as one simulation. Play can override this budget for an MCTS category; greedy networks and baselines ignore it.'],
    [`c_puct · ${settings.c_puct}`, 'Exploration constant: weights policy prior and uncertainty against observed value. The code selects the largest score = Q + c_puct · P · √(N_parent + 1) / (1 + N_child), where N_parent is the sum of child visits.'],
    [`FPU reduction · ${settings.fpu_reduction}`, 'First-play urgency: unvisited moves use Q = parent network value − reduction, instead of a child’s observed mean value. This lowers their starting Q, favouring moves with supporting evidence while the prior-based exploration term still encourages exploration.'],
  ];
  for (const [term, meaning] of entries) glossary.append(element('dt', '', term), element('dd', '', meaning));
  section.append(glossary);
  return section;
}
