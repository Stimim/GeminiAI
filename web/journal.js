'use strict';

(() => {
  const index = document.getElementById('journal-index');
  const reader = document.getElementById('journal-reader');
  const body = document.getElementById('article-body');
  const title = document.getElementById('article-title');
  const status = document.getElementById('article-status');
  const navigation = document.getElementById('chapter-navigation');
  const cards = [...document.querySelectorAll('.journal-card')];
  const chapters = cards.map(card => ({
    slug: card.querySelector('h2 a').hash.slice(1),
    title: card.querySelector('h2').textContent,
    file: card.dataset.file,
    meta: card.querySelector('.journal-meta'),
    teaser: card.querySelector('.journal-teaser').textContent,
    number: card.querySelector('.eyebrow').textContent,
  }));
  let request = 0;
  let controller;

  function chapterLink(chapter, label) {
    const link = document.createElement('a');
    link.href = chapter ? `#${chapter.slug}` : '#';
    const direction = document.createElement('span');
    direction.className = 'eyebrow';
    direction.textContent = label;
    const name = document.createElement('strong');
    name.textContent = chapter ? chapter.title : 'All chapters';
    link.append(direction, name);
    return link;
  }

  async function render(focus) {
    const slug = location.hash.slice(1);
    // Preserve the current view when the keyboard skip link targets main.
    if (slug === 'content') return;
    const current = ++request;
    controller?.abort();
    if (!slug) {
      reader.hidden = true;
      index.hidden = false;
      document.title = 'Journal · Minesweeper Flags';
      if (focus) {
        document.getElementById('journal-title').focus({preventScroll: true});
        window.scrollTo(0, 0);
      }
      return;
    }
    const position = chapters.findIndex(chapter => chapter.slug === slug);
    const chapter = chapters[position];
    index.hidden = true;
    reader.hidden = false;
    body.replaceChildren();
    navigation.replaceChildren();
    title.textContent = chapter?.title || 'Chapter not found';
    document.getElementById('article-number').textContent = chapter?.number || 'A wrong turn';
    document.getElementById('article-meta').replaceChildren(...(chapter ? [...chapter.meta.childNodes].map(node => node.cloneNode(true)) : []));
    document.getElementById('article-teaser').textContent = chapter?.teaser || 'This chapter does not exist. Return to all chapters to pick a story.';
    document.title = `${title.textContent} · Journal · Minesweeper Flags`;
    status.textContent = chapter ? 'Loading chapter…' : '';
    title.focus({preventScroll: true});
    window.scrollTo(0, 0);
    if (!chapter) return;
    controller = new AbortController();
    try {
      // The file path comes only from our static index, never the URL hash.
      const response = await fetch(`/static/journal/${chapter.file}`, {signal: controller.signal});
      if (!response.ok) throw new Error('Article unavailable');
      const source = await response.text();
      if (current !== request) return;
      // Only repository-owned HTML fragments are parsed. No user content enters this document.
      const parsed = new DOMParser().parseFromString(source, 'text/html');
      body.replaceChildren(...[...parsed.body.childNodes].map(node => document.importNode(node, true)));
      status.textContent = '';
      navigation.append(
        chapterLink(chapters[position - 1], '← Previous'),
        chapterLink(chapters[position + 1], 'Next →'),
      );
    } catch (error) {
      if (current !== request || error.name === 'AbortError') return;
      status.textContent = 'This chapter could not be loaded. Check your connection and try again.';
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = 'Try again';
      retry.addEventListener('click', () => render(false), {once: true});
      body.append(retry);
    }
  }

  window.addEventListener('hashchange', () => render(true));
  render(false);
})();
