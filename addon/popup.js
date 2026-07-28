const GROUP_ID_NONE = browser.tabGroups?.TAB_GROUP_ID_NONE ?? -1;

// Anything shorter mangles unrelated words when replaced as a substring.
const MIN_REDACT_LENGTH = 4;

const state = { tabs: [], groups: [] };

function redact(text, words) {
  let result = text;
  for (const word of words) {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    result = result.replace(new RegExp(escaped, "gi"), "*REDACTED*");
  }
  return result;
}

function isWebContent(url) {
  try {
    return /^https?:$/.test(new URL(url).protocol);
  } catch (e) {
    return false;
  }
}

function tabsByGroup() {
  const map = new Map();
  for (const tab of state.tabs) {
    const key = tab.groupId ?? GROUP_ID_NONE;
    if (!map.has(key)) {
      map.set(key, []);
    }
    map.get(key).push(tab);
  }
  return map;
}

/**
 * Shell shared by the group and ungrouped sections: a header row followed by
 * one line per tab.
 *
 * @param {object} options
 * @param {string} options.className
 * @param {HTMLInputElement} options.checkbox controls whether the section is exported
 * @param {HTMLElement} options.caption editable name field, or a static label
 * @param {HTMLElement} [options.swatch] group colour dot
 * @param {object[]} options.tabs
 */
function buildSection({ className, checkbox, caption, swatch, tabs }) {
  const section = document.createElement("div");
  section.className = className;

  const head = document.createElement("div");
  head.className = "group_head";

  const count = document.createElement("span");
  count.className = "count";
  count.textContent = `${tabs.length} tabs`;

  head.append(checkbox);
  if (swatch) {
    head.append(swatch);
  }
  head.append(caption, count);
  section.append(head);

  for (const tab of tabs) {
    const title = document.createElement("div");
    title.className = "tab_title";
    title.textContent = tab.title || tab.url;
    section.append(title);
  }
  return section;
}

function buildGroupSection(group, tabs) {
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = true;
  checkbox.dataset.groupId = String(group.id);

  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = group.color || "gray";

  const caption = document.createElement("input");
  caption.type = "text";
  caption.className = "group_name";
  caption.value = group.title || "";
  caption.dataset.nameFor = String(group.id);
  if (!group.title) {
    caption.classList.add("unnamed");
    caption.placeholder = "Unnamed group - please add a name";
  }

  return buildSection({ className: "group", checkbox, caption, swatch, tabs });
}

function buildUngroupedSection(tabs) {
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.id = "include_ungrouped";
  checkbox.checked = true;

  const caption = document.createElement("label");
  caption.className = "group_name";
  caption.htmlFor = "include_ungrouped";
  caption.textContent = "Ungrouped - not part of any group";

  return buildSection({
    className: "group ungrouped",
    checkbox,
    caption,
    tabs,
  });
}

function render() {
  const root = document.getElementById("groups");
  root.replaceChildren();

  const grouped = tabsByGroup();
  const named = state.groups.filter(g => (grouped.get(g.id) || []).length);

  if (!named.length) {
    const empty = document.createElement("p");
    empty.textContent =
      "No tab groups found. Create and name your groups first, then reopen this popup.";
    root.append(empty);
  }

  for (const group of named) {
    root.append(buildGroupSection(group, grouped.get(group.id)));
  }

  const ungrouped = grouped.get(GROUP_ID_NONE) || [];
  if (ungrouped.length) {
    root.append(buildUngroupedSection(ungrouped));
  }
}

function collect() {
  const includeUngrouped = !!document.getElementById("include_ungrouped")
    ?.checked;
  const words = document
    .getElementById("remove_words")
    .value.split(/\s+/)
    .filter(word => word.length >= MIN_REDACT_LENGTH);

  const picked = new Set();
  const titles = {};
  for (const checkbox of document.querySelectorAll("input[data-group-id]")) {
    if (checkbox.checked) {
      const id = Number(checkbox.dataset.groupId);
      picked.add(id);
      titles[id] = document
        .querySelector(`input[data-name-for="${id}"]`)
        .value.trim();
    }
  }

  const grouped = tabsByGroup();
  const selected = [];
  for (const [groupId, tabs] of grouped) {
    if (groupId === GROUP_ID_NONE ? includeUngrouped : picked.has(groupId)) {
      selected.push(...tabs);
    }
  }

  const tabList = selected.map(tab => {
    // Ungrouped tabs are not a cluster: a null groupId plus an explicit flag so
    // no consumer can accidentally treat them as one shared group.
    const isGrouped = (tab.groupId ?? GROUP_ID_NONE) !== GROUP_ID_NONE;
    return {
      id: tab.id,
      windowId: tab.windowId,
      groupId: isGrouped ? tab.groupId : null,
      grouped: isGrouped,
      title: redact(tab.title || "", words),
      url: tab.url,
      pinned: tab.pinned,
      lastAccessed: tab.lastAccessed,
    };
  });

  return {
    tab_list: tabList,
    group_titles: titles,
    groups: state.groups
      .filter(group => picked.has(group.id))
      .map(({ id, color, collapsed, windowId }) => ({
        id,
        title: titles[id],
        color,
        collapsed,
        windowId,
      })),
  };
}

async function copyData() {
  const status = document.getElementById("status");
  try {
    await navigator.clipboard.writeText(JSON.stringify(collect()));
    status.textContent = "Copied. Paste it into the form.";
  } catch (e) {
    status.textContent = `Copy failed: ${e.message}`;
  }
}

async function init() {
  document.getElementById("copyButton").addEventListener("click", copyData);

  try {
    const [tabs, groups] = await Promise.all([
      browser.tabs.query({}),
      browser.tabGroups.query({}),
    ]);

    // Only real web content, mirroring what Firefox itself feeds to clustering.
    state.tabs = tabs.filter(tab => !tab.incognito && isWebContent(tab.url));
    state.groups = groups;
    render();
  } catch (e) {
    document.getElementById("status").textContent =
      `Could not read your tabs: ${e.message}`;
  }
}

document.addEventListener("DOMContentLoaded", init);
