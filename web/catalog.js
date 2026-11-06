'use strict';

window.AgentCatalog = (() => {
  const neural = category => category && !['random', 'solver'].includes(category.id);
  const generations = category => [...(category?.generations || [])].sort((a, b) => Number(b.generation ?? 0) - Number(a.generation ?? 0) || Number(b.training?.bootstrap ?? 0) - Number(a.training?.bootstrap ?? 0) || b.id.localeCompare(a.id));
  const categoryFor = (categories, id) => categories.find(category => category.id === id || category.generations.some(generation => generation.id === id));
  const resolve = (category, value) => value || category?.default_generation || '';
  function fillGenerations(select, category) {
    select.replaceChildren();
    select.hidden = !neural(category);
    select.disabled = !neural(category) || !category?.available;
    if (!neural(category)) return;
    select.add(new Option('Latest available', ''));
    for (const generation of generations(category)) {
      const option = new Option(generation.label || `Generation ${generation.generation}`, generation.id);
      option.disabled = !generation.available;
      select.add(option);
    }
  }
  function selected(categorySelect, generationSelect, categories) {
    if (categorySelect.value === 'human') return 'human';
    const category = categories.find(item => item.id === categorySelect.value);
    return resolve(category, generationSelect.value);
  }
  function setSelection(categorySelect, generationSelect, categories, id) {
    const category = categoryFor(categories, id);
    if (!category) return;
    categorySelect.value = category.id;
    fillGenerations(generationSelect, category);
    if (id !== category.id && [...generationSelect.options].some(option => option.value === id)) generationSelect.value = id;
  }
  const elo = entry => entry?.games !== 0 && Number.isFinite(entry?.elo ?? entry?.rating) && Number.isFinite(entry?.uncertainty) ? `${Math.round(entry.elo ?? entry.rating)} ± ${Math.round(entry.uncertainty)}` : 'Unrated';
  function training(training) {
    if (!training || (training.bootstrap === 0 && training.selfplay === 0)) return 'No training (baseline)';
    const parts = [];
    for (const [stage, label] of [['bootstrap', 'Bootstrap'], ['selfplay', 'Self-play']]) {
      const value = training[stage];
      if (value === null || value === undefined) continue;
      if (typeof value !== 'object') { parts.push(`${label}: ${Number(value).toLocaleString()} ${stage === 'selfplay' ? 'iterations' : 'steps'}`); continue; }
      const metrics = [];
      for (const [key, unit] of [['completed_steps', 'steps'], ['steps', 'steps'], ['positions', 'positions'], ['generated_selfplay_games', 'games'], ['games', 'games']]) {
        if (Number.isFinite(value[key]) && !metrics.some(metric => metric.endsWith(` ${unit}`))) metrics.push(`${value[key].toLocaleString()} ${unit}`);
      }
      parts.push(`${label}: ${metrics.length ? metrics.join(', ') : 'not recorded'}`);
    }
    return parts.join(' · ') || 'No training recorded';
  }
  return {neural, generations, categoryFor, resolve, fillGenerations, selected, setSelection, elo, training};
})();
