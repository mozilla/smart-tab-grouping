# Tab Group Collector

A small Firefox add-on that exports your tab groups as JSON, so they can be used as
reference data for evaluating automatic tab grouping.

Firefox groups tabs on device, using a local embedding model to cluster them and a
local text model to name each cluster. Evaluating that requires examples of how people
genuinely organise their own tabs. This add-on collects those examples: you arrange and
name your groups, review what is about to leave your browser, and copy it out.

Requires Firefox 139 or newer, which is when the
[`tabGroups`](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabGroups)
extension API shipped. Any channel works.

## Installing

The add-on is unsigned and meant to be installed temporarily:

1. Open `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on...**
3. Select `manifest.json` from this directory, or a built `.xpi`

It is removed automatically when Firefox restarts.

## Using it

1. Arrange your open tabs into groups and name them the way you actually want them.
2. Open the add-on from the toolbar.
3. Untick any group you would rather not share, and name any group outlined in red.
4. Optionally list words to redact from tab titles.
5. Click **Copy JSON**.

Nothing is uploaded. The JSON goes to your clipboard and it is up to you what you do
with it. Paste it into a text editor first if you want to read it.

## What is collected

For each tab in a selected group: title, URL, pinned state, last-used time, and the tab,
group and window identifiers. For each group: name, colour, collapsed state and window.

Not collected: page contents, cookies, passwords, form data, browsing history, private
window tabs, and non-web pages such as `about:`, `chrome:` and `file:`. Only `http` and
`https` tabs are eligible, which mirrors what Firefox itself considers for grouping.

Titles and URLs are exported in full and untruncated. The tab title in particular is
the text the clustering model reads, so truncating it would make the data less useful.

## Output format

```json
{
  "tab_list": [
    {"id": 3, "windowId": 59, "groupId": 12, "grouped": true,
     "title": "...", "url": "https://...",
     "pinned": false, "lastAccessed": 1729090648093},
    {"id": 9, "windowId": 59, "groupId": null, "grouped": false,
     "title": "...", "url": "https://...",
     "pinned": false, "lastAccessed": 1729090812004}
  ],
  "group_titles": {"12": "Marathon training"},
  "groups": [{"id": 12, "title": "Marathon training", "color": "blue",
              "collapsed": false, "windowId": 59}]
}
```

Group membership is `groupId`. `windowId` is recorded separately because Firefox
considers each window on its own when it groups tabs.

### Ungrouped tabs

Ungrouped tabs are included by default and marked two ways: `groupId` is `null` and
`grouped` is `false`. They appear in `tab_list` but never in `group_titles` or `groups`.

**They are not a cluster.** They are tabs the user deliberately left out of every group,
so they are unlabeled rather than related to each other. Grouping `tab_list` by
`groupId` without filtering collapses them all into one bogus `null` cluster and will
badly distort any partition metric. Choose a semantics explicitly:

- **noise / abstention** - exclude them from the partition metric and score them
  separately, as tabs the model should have left alone
- **singletons** - give each one its own cluster id

The two give substantially different numbers, so results should state which was used.

## Building

With [web-ext](https://github.com/mozilla/web-ext):

```sh
web-ext lint
web-ext build
```
