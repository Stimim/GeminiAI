'use strict';

const byId = id => document.getElementById(id);
let detailRequest = 0;
let agentList = [];
let activeAgentId = '';

function navigateAgent(direction) {
  const available = agentList.filter(agent => agent.available);
  if (!available.length) return;
  const current = available.findIndex(agent => agent.id === activeAgentId);
  const index = current < 0 ? 0 : (current + direction + available.length) % available.length;
  location.hash = encodeURIComponent(available[index].id);
}

function renderSwitcher(currentId) {
  const root = byId('agent-switcher');
  const strip = element('nav', 'agent-switcher-strip');
  strip.setAttribute('aria-label', 'Switch agent');
  const available = agentList.filter(agent => agent.available);
  agentList.forEach(agent => {
    const chip = element(agent.available ? 'a' : 'span', `agent-chip${agent.available ? '' : ' is-disabled'}${agent.id === currentId ? ' is-current' : ''}`);
    if (agent.available) chip.href = `#${encodeURIComponent(agent.id)}`;
    else { chip.setAttribute('aria-disabled', 'true'); chip.title = 'Unavailable'; }
    if (agent.id === currentId) chip.setAttribute('aria-current', 'page');
    chip.append(element('span', 'agent-chip-name', agent.name), element('span', 'badge', agent.kind));
    strip.append(chip);
  });
  const controls = element('div', 'agent-switcher-controls');
  const previous = element('button', 'secondary', '← Previous');
  previous.type = 'button';
  previous.addEventListener('click', () => navigateAgent(-1));
  const next = element('button', 'secondary', 'Next →');
  next.type = 'button';
  next.addEventListener('click', () => navigateAgent(1));
  controls.append(previous, next);
  root.replaceChildren(strip, controls);
}

document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.target instanceof Element && event.target.closest('input, select, textarea, [contenteditable="true"]')) return;
  if (!activeAgentId) return;
  if (event.key === 'ArrowLeft') { event.preventDefault(); navigateAgent(-1); }
  else if (event.key === 'ArrowRight') { event.preventDefault(); navigateAgent(1); }
});


async function api(path) {
  const response = await fetch(path, {headers: {Accept: 'application/json'}});
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const detail = data.detail || data.error || response.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}


function structuredSection(title, value) {
  const section = element('section', 'detail-section');
  section.append(element('h3', '', title));
  if (value === null || value === undefined || (typeof value === 'object' && !Object.keys(value).length)) {
    section.append(element('p', 'muted', 'Not applicable or not provided for this agent.'));
  } else if (typeof value === 'object' && !Array.isArray(value)) {
    const list = element('dl', 'metadata');
    for (const [key, item] of Object.entries(value)) {
      list.append(element('dt', '', key.replaceAll('_', ' ')));
      const description = element('dd');
      if (key === 'training' && Array.isArray(item)) {
        const runs = element('ol');
        for (const run of item) {
          const row = element('li');
          const count = number => Number(number).toLocaleString();
          row.append(element('strong', '', run.stage === 'bootstrap' ? 'Solver distillation' : 'Reinforcement self-play'));
          row.append(element('p', '', `${count(run.completed_steps)} completed optimizer steps; ${count(run.steps_per_iteration)} steps per iteration.`));
          if (run.positions !== null) row.append(element('p', '', `${count(run.positions)} exact-labelled training positions.`));
          if (run.generated_selfplay_games) row.append(element('p', '', `${count(run.generated_selfplay_games)} generated games across ${count(run.requested_iterations)} requested iterations.`));
          if (run.data) row.append(element('p', '', `Dataset: ${run.data}`));
          runs.append(row);
        }
        description.append(runs);
      } else {
        description.append(element(typeof item === 'object' && item !== null ? 'pre' : 'span', 'metadata-value', typeof item === 'object' && item !== null ? JSON.stringify(item, null, 2) : String(item ?? 'Not provided')));
      }
      list.append(description);
    }
    section.append(list);
  } else {
    section.append(element('pre', 'metadata-value', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
  }
  return section;
}

function showError(message) {
  byId('catalog-error').textContent = message;
  byId('catalog-error').hidden = !message;
}

async function showDetail() {
  const request = ++detailRequest;
  const raw = location.hash.slice(1);
  const hasDetail = Boolean(raw);
  byId('catalog').hidden = hasDetail;
  byId('agent-detail').hidden = !hasDetail;
  showError('');
  const content = byId('detail-content');
  if (!hasDetail) { activeAgentId = ''; byId('agent-switcher').replaceChildren(); document.title = 'AIs · Minesweeper Flags'; return; }
  let id;
  try { id = decodeURIComponent(raw); } catch { id = raw; }
  if (activeAgentId && activeAgentId !== id) window.scrollTo({top: 0, behavior: 'instant'});
  activeAgentId = id;
  renderSwitcher(id);
  content.replaceChildren(element('h2', '', 'Loading agent…'));
  try {
    const agent = await api(`/api/agents/${encodeURIComponent(id)}`);
    if (request !== detailRequest) return;
    document.title = `${agent.name} · AIs · Minesweeper Flags`;
    const heading = element('div', 'detail-heading');
    const title = element('h2', '', agent.name);
    title.id = 'detail-title';
    heading.append(title, element('span', 'badge', agent.kind), element('span', `badge ${agent.available ? 'available' : 'unavailable'}`, agent.available ? 'Available' : 'Unavailable'));
    const explanation = element('section', 'detail-section');
    explanation.append(element('h3', '', 'How it works'), element('div', 'explanation', agent.explanation || 'No explanation provided.'));
    const architecture = ['net', 'mcts'].includes(agent.kind) ? architectureSection(agent) : structuredSection('Architecture / shape', agent.shape);
    content.replaceChildren(heading, element('p', 'lede', agent.summary), architecture, explanation, structuredSection('Provenance', agent.provenance), structuredSection('Settings', agent.settings));
    if (agent.checkpoint) content.append(structuredSection('Checkpoint', agent.checkpoint));
    if (agent.available) {
      const link = element('a', 'button', 'Compare in the arena →');
      link.href = `/arena.html?a=${encodeURIComponent(agent.id)}`;
      content.append(link);
    } else {
      content.append(element('p', 'notice', 'This agent is listed for reference but is not currently available to play.'));
    }
    byId('agent-detail').focus({preventScroll: true});
  } catch (error) {
    if (request !== detailRequest) return;
    content.replaceChildren(element('h2', '', 'Agent unavailable'));
    showError(`Could not load this agent: ${error.message} Use “All agents” to return to the catalog.`);
  }
}

async function loadCatalog() {
  try {
    const data = await api('/api/agents');
    const agents = Array.isArray(data) ? data : data.agents;
    if (!Array.isArray(agents)) throw new Error('The server did not return an agent list.');
    agentList = agents;
    byId('agent-count').textContent = `${agents.length} agent${agents.length === 1 ? '' : 's'}`;
    byId('catalog-status').textContent = agents.length ? '' : 'No agents are registered yet.';
    const cards = agents.map(agent => {
      const card = element('article', 'surface agent-card');
      const badges = element('div', 'card-badges');
      badges.append(element('span', 'badge', agent.kind), element('span', `availability ${agent.available ? 'available' : 'unavailable'}`, agent.available ? 'Available' : 'Unavailable'));
      const heading = element('h3');
      const link = element('a', '', agent.name);
      link.href = `#${encodeURIComponent(agent.id)}`;
      heading.append(link);
      const more = element('a', 'text-link', 'Explore agent →');
      more.href = link.href;
      more.setAttribute('aria-label', `Explore ${agent.name}`);
      card.append(badges, heading, element('p', '', agent.summary), more);
      return card;
    });
    byId('agent-grid').replaceChildren(...cards);
  } catch (error) {
    byId('catalog-status').textContent = 'The catalog could not be loaded. Reload the page to try again.';
    if (!location.hash) showError(error.message);
  }
}

window.addEventListener('hashchange', showDetail);
loadCatalog().then(showDetail);
