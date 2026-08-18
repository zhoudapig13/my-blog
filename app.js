const THEME_OPTIONS = ["journal", "rose", "atelier", "garden", "brownrose", "nightlibrary"];
const previewTheme = new URLSearchParams(location.search).get("theme");
const initialTheme = THEME_OPTIONS.includes(previewTheme)
  ? previewTheme
  : THEME_OPTIONS[Math.floor(Math.random() * THEME_OPTIONS.length)];
document.body.dataset.theme = initialTheme;

let state = {
  theme: initialTheme,
  profile: {},
  posts: [],
  diary: [],
  resources: [],
  friends: [],
  plans: []
};

const FILTER_ALL = "全部";
const GITHUB_OWNER = "zhoudapig13";
const GITHUB_REPO = "my-blog";
const GITHUB_BRANCH = "main";
const HOME_PINNED_POST_ID = "llm-loss-optimizers-gradients";
const TOKEN_STORAGE_KEY = "my-blog.github-token";
const WRITER_SETTINGS_DB = "my-blog-writer-settings";
const OBSIDIAN_DIRECTORY_KEY = "obsidian-directory";
const POST_REDIRECTS = {
  "未命名文章": "llm-loss-optimizers-gradients"
};
let activeFilter = FILTER_ALL;
let resourcePreviewZoom = 100;
let resourcePreviewMode = "frame";
let resourcePreviewBaseUrl = "";
let obsidianDirectoryHandle = null;
let obsidianImageIndex = null;
let writerImageUploadInProgress = false;
let writerPreviewTimer = null;
let writerPreviewIdleHandle = null;
let writerMathIdleHandle = null;
let writerPreviewRevision = 0;
let writerState = {
  token: sessionStorage.getItem(TOKEN_STORAGE_KEY) || "",
  user: null,
  files: [],
  currentPath: "",
  currentSha: ""
};

const routeMap = {
  home: document.querySelector("#homeView"),
  blog: document.querySelector("#blogView"),
  diary: document.querySelector("#diaryView"),
  post: document.querySelector("#postView"),
  resources: document.querySelector("#resourcesView"),
  plan: document.querySelector("#planView"),
  writer: document.querySelector("#writerView")
};

const els = {
  body: document.body,
  search: document.querySelector("#searchInput"),
  profileImage: document.querySelector("#profileImage"),
  profileName: document.querySelector("#profileName"),
  profileBio: document.querySelector("#profileBio"),
  recentPosts: document.querySelector("#recentPosts"),
  blogArchive: document.querySelector("#blogArchive"),
  postList: document.querySelector("#postList"),
  diaryList: document.querySelector("#diaryList"),
  postReader: document.querySelector("#postReader"),
  relatedPosts: document.querySelector("#relatedPosts"),
  articleToc: document.querySelector("#articleToc"),
  readerLayout: document.querySelector("#readerLayout"),
  categoryList: document.querySelector("#categoryList"),
  tagList: document.querySelector("#tagList"),
  friendList: document.querySelector("#friendList"),
  categoryCount: document.querySelector("#categoryCount"),
  tagCount: document.querySelector("#tagCount"),
  friendCount: document.querySelector("#friendCount"),
  blogFilters: document.querySelector("#blogFilters"),
  resourceList: document.querySelector("#resourceList"),
  resourcePreviewDialog: document.querySelector("#resourcePreviewDialog"),
  resourcePreviewTitle: document.querySelector("#resourcePreviewTitle"),
  resourcePreviewMeta: document.querySelector("#resourcePreviewMeta"),
  resourcePreviewShell: document.querySelector(".resource-preview-shell"),
  resourcePreviewFrame: document.querySelector("#resourcePreviewFrame"),
  resourcePreviewCode: document.querySelector("#resourcePreviewCode"),
  resourcePreviewDownload: document.querySelector("#resourcePreviewDownload"),
  resourcePreviewOpen: document.querySelector("#resourcePreviewOpen"),
  resourcePreviewZoomOut: document.querySelector("#resourcePreviewZoomOut"),
  resourcePreviewZoomReset: document.querySelector("#resourcePreviewZoomReset"),
  resourcePreviewZoomIn: document.querySelector("#resourcePreviewZoomIn"),
  resourcePreviewFullscreen: document.querySelector("#resourcePreviewFullscreen"),
  resourcePreviewClose: document.querySelector("#resourcePreviewClose"),
  imagePreviewDialog: document.querySelector("#imagePreviewDialog"),
  imagePreviewImage: document.querySelector("#imagePreviewImage"),
  imagePreviewTitle: document.querySelector("#imagePreviewTitle"),
  imagePreviewClose: document.querySelector("#imagePreviewClose"),
  planList: document.querySelector("#planList"),
  planOwnerPanel: document.querySelector("#planOwnerPanel"),
  progressValue: document.querySelector("#progressValue"),
  writerNav: document.querySelector("#writerNav"),
  writerLoginPanel: document.querySelector("#writerLoginPanel"),
  writerStudio: document.querySelector("#writerStudio"),
  githubTokenInput: document.querySelector("#githubTokenInput"),
  githubLoginButton: document.querySelector("#githubLoginButton"),
  githubLogoutButton: document.querySelector("#githubLogoutButton"),
  writerLoginMessage: document.querySelector("#writerLoginMessage"),
  writerAuthStatus: document.querySelector("#writerAuthStatus"),
  writerPostSelect: document.querySelector("#writerPostSelect"),
  writerTitle: document.querySelector("#writerTitle"),
  writerDate: document.querySelector("#writerDate"),
  writerCategory: document.querySelector("#writerCategory"),
  writerTags: document.querySelector("#writerTags"),
  writerPdf: document.querySelector("#writerPdf"),
  writerPdfTitle: document.querySelector("#writerPdfTitle"),
  writerSummary: document.querySelector("#writerSummary"),
  writerSummaryCount: document.querySelector("#writerSummaryCount"),
  writerSummaryPreview: document.querySelector("#writerSummaryPreview"),
  writerContent: document.querySelector("#writerContent"),
  writerObsidianImagesButton: document.querySelector("#writerObsidianImagesButton"),
  writerObsidianImagesInput: document.querySelector("#writerObsidianImagesInput"),
  writerObsidianStatus: document.querySelector("#writerObsidianStatus"),
  writerPreview: document.querySelector("#writerPreview"),
  writerPreviewStatus: document.querySelector("#writerPreviewStatus"),
  writerPreviewRefresh: document.querySelector("#writerPreviewRefresh"),
  writerCurrentPath: document.querySelector("#writerCurrentPath"),
  writerNewButton: document.querySelector("#writerNewButton"),
  writerSaveButton: document.querySelector("#writerSaveButton"),
  writerSaveMessage: document.querySelector("#writerSaveMessage"),
  postForm: document.querySelector("#postForm"),
  markdownFile: document.querySelector("#markdownFile"),
  livePreview: document.querySelector("#livePreview"),
  downloadDraft: document.querySelector("#downloadDraft")
};

async function loadSiteData() {
  try {
    const response = await fetch(`data/site.json?v=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state = {
      ...state,
      ...data,
      theme: THEME_OPTIONS.includes(previewTheme)
        ? previewTheme
        : state.theme
    };
  } catch (error) {
    document.querySelector("main").insertAdjacentHTML(
      "afterbegin",
      '<div class="empty-state">没有读取到 <code>data/site.json</code>。请先运行 <code>node scripts/build-site-data.js</code>，或通过本地服务器打开网站。</div>'
    );
    console.error(error);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function slugify(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "") || `post-${Date.now()}`;
}

function headingSlug(value, counts = new Map()) {
  const base =
    String(value)
      .replace(/&[#a-z0-9]+;/gi, " ")
      .replace(/[*_`[\]$]/g, "")
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, "-")
      .replace(/^-+|-+$/g, "") || "section";
  const count = counts.get(base) || 0;
  counts.set(base, count + 1);
  return count ? `${base}-${count + 1}` : base;
}

function excerpt(content, length = 110) {
  const plain = content
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/\$([^$\n]+)\$/g, "$1")
    .replace(/[#>*_`\-[\]()]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return plain.length > length ? `${plain.slice(0, length)}...` : plain;
}

function markdownToHtml(markdown) {
  const blocks = [];
  let text = String(markdown || "").replace(/\r\n/g, "\n");

  text = text.replace(/^---\n[\s\S]*?\n---\n?/, "");

  text = text.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, language, code) => {
    const token = `@@BLOCK${blocks.length}@@`;
    blocks.push(renderCodeBlock(code, language));
    return token;
  });

  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    const token = `@@BLOCK${blocks.length}@@`;
    blocks.push(`<div class="math-block">\\[${escapeHtml(cleanMath(math))}\\]</div>`);
    return token;
  });

  text = escapeHtml(text);

  const lines = text.split("\n");
  const html = [];
  const headingCounts = new Map();
  let listOpen = false;

  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const rawLine = lines[lineIndex];
    const line = rawLine.trim();

    if (!line) {
      closeList();
      continue;
    }

    if (line.startsWith("@@BLOCK")) {
      closeList();
      html.push(line);
      continue;
    }

    if (isMarkdownTableRow(line) && isMarkdownTableSeparator(lines[lineIndex + 1]?.trim() || "")) {
      closeList();
      const headers = splitMarkdownTableRow(line);
      const alignments = splitMarkdownTableRow(lines[lineIndex + 1].trim()).map((cell) => {
        const left = cell.startsWith(":");
        const right = cell.endsWith(":");
        if (left && right) return "center";
        if (right) return "right";
        return "left";
      });
      const rows = [];
      lineIndex += 2;
      while (lineIndex < lines.length && isMarkdownTableRow(lines[lineIndex].trim())) {
        rows.push(splitMarkdownTableRow(lines[lineIndex].trim()));
        lineIndex += 1;
      }
      lineIndex -= 1;
      html.push(renderMarkdownTable(headers, alignments, rows));
      continue;
    }

    if (/^#{1,4}\s/.test(line)) {
      closeList();
      const level = line.match(/^#+/)[0].length;
      const headingText = line.replace(/^#{1,4}\s/, "");
      const id = headingSlug(headingText, headingCounts);
      html.push(`<h${level} id="${escapeHtml(id)}">${inlineMarkdown(headingText)}</h${level}>`);
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>`);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineMarkdown(line.replace(/^\d+\.\s+/, ""))}</li>`);
      continue;
    }

    if (/^&gt;\s?\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/i.test(line)) {
      closeList();
      const label = line.replace(/^&gt;\s?\[!(\w+)\]\s*/i, "$1");
      html.push(`<blockquote class="callout"><strong>${label}</strong></blockquote>`);
      continue;
    }

    if (/^&gt;\s?/.test(line)) {
      closeList();
      html.push(`<blockquote>${inlineMarkdown(line.replace(/^&gt;\s?/, ""))}</blockquote>`);
      continue;
    }

    closeList();
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }

  closeList();

  return html.join("\n").replace(/@@BLOCK(\d+)@@/g, (_, index) => blocks[Number(index)]);
}

function isMarkdownTableRow(line) {
  return line.includes("|") && splitMarkdownTableRow(line).length > 1;
}

function isMarkdownTableSeparator(line) {
  const cells = splitMarkdownTableRow(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function splitMarkdownTableRow(line) {
  const trimmed = String(line || "").trim();
  const withoutEdges = trimmed.replace(/^\|/, "").replace(/\|$/, "");
  return withoutEdges.split("|").map((cell) => cell.trim());
}

function renderMarkdownTable(headers, alignments, rows) {
  const alignAttr = (index) => ` style="text-align: ${alignments[index] || "left"}"`;
  const headerHtml = headers
    .map((header, index) => `<th${alignAttr(index)}>${inlineMarkdown(header)}</th>`)
    .join("");
  const rowsHtml = rows
    .map((row) => {
      const cells = headers.map((_, index) => row[index] || "");
      return `<tr>${cells.map((cell, index) => `<td${alignAttr(index)}>${inlineMarkdown(cell)}</td>`).join("")}</tr>`;
    })
    .join("");

  return `
    <div class="markdown-table-scroll">
      <table>
        <thead><tr>${headerHtml}</tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
  `;
}

function inlineMarkdown(value) {
  return value
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, source) => {
      if (isLocalImageReference(source)) {
        return `<span class="obsidian-image-placeholder">本地图片待上传：${alt || imageReferenceBaseName(source)}</span>`;
      }
      const safeSource = escapeHtml(source);
      const safeAlt = escapeHtml(alt || "文章图片");
      return `
        <span class="article-image-frame">
          <img src="${safeSource}" alt="${safeAlt}" loading="lazy" />
          <button
            class="image-zoom-button"
            type="button"
            data-image-preview
            data-image-src="${safeSource}"
            data-image-alt="${safeAlt}"
            aria-label="放大浏览${safeAlt}"
            title="放大浏览"
          >＋</button>
        </span>
      `;
    })
    .replace(/!\[\[([^\]\n]+\.(?:png|jpe?g|gif|webp|svg|bmp|avif))(?:\|([^\]]+))?\]\]/gi, (_, source, option) => {
      const fileName = source.trim().split(/[\\/]/).pop();
      const optionText = String(option || "").trim();
      const label = optionText && !/^\d+(?:x\d+)?$/i.test(optionText) ? optionText : fileName;
      return `<span class="obsidian-image-placeholder">Obsidian 图片待上传：${label}</span>`;
    })
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[\[([^\]]+)\]\]/g, '<span class="wiki-link">$1</span>')
    .replace(/\$([^$\n]+)\$/g, (_, math) => `<span class="math-inline">\\(${cleanMath(math)}\\)</span>`)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function cleanMath(value) {
  return String(value).trim().replace(/\\([*_])/g, "$1");
}

const PYTHON_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "case", "class", "continue",
  "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
  "if", "import", "in", "is", "lambda", "match", "nonlocal", "not", "or",
  "pass", "raise", "return", "try", "while", "with", "yield"
]);

const PYTHON_BUILTINS = new Set([
  "abs", "all", "any", "bool", "bytes", "callable", "dict", "enumerate", "filter",
  "float", "format", "frozenset", "getattr", "hasattr", "hash", "help", "hex",
  "id", "input", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
  "max", "min", "next", "object", "open", "ord", "pow", "print", "property",
  "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
  "super", "tuple", "type", "vars", "zip"
]);

function highlightPython(source) {
  const code = String(source || "");
  let html = "";
  let index = 0;
  let expectDefinition = false;
  const token = (className, value) => `<span class="syntax-${className}">${escapeHtml(value)}</span>`;

  while (index < code.length) {
    const rest = code.slice(index);

    if (rest[0] === "#") {
      const end = code.indexOf("\n", index);
      const stop = end === -1 ? code.length : end;
      html += token("comment", code.slice(index, stop));
      index = stop;
      continue;
    }

    if (rest.startsWith('"""') || rest.startsWith("'''") || rest[0] === '"' || rest[0] === "'") {
      const delimiter = rest.startsWith('"""') || rest.startsWith("'''") ? rest.slice(0, 3) : rest[0];
      let end = index + delimiter.length;
      while (end < code.length) {
        if (code.startsWith(delimiter, end)) {
          end += delimiter.length;
          break;
        }
        if (code[end] === "\\") end += 1;
        end += 1;
      }
      html += token("string", code.slice(index, end));
      index = end;
      expectDefinition = false;
      continue;
    }

    const decorator = rest.match(/^@[A-Za-z_][\w.]*/);
    if (decorator) {
      html += token("decorator", decorator[0]);
      index += decorator[0].length;
      continue;
    }

    const number = rest.match(/^(?:0[xob][\da-f_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:e[+-]?\d[\d_]*)?)/i);
    if (number) {
      html += token("number", number[0]);
      index += number[0].length;
      expectDefinition = false;
      continue;
    }

    const identifier = rest.match(/^[A-Za-z_]\w*/);
    if (identifier) {
      const value = identifier[0];
      if (PYTHON_KEYWORDS.has(value)) {
        html += token("keyword", value);
        expectDefinition = value === "def" || value === "class";
      } else if (expectDefinition) {
        html += token("function", value);
        expectDefinition = false;
      } else if (["True", "False", "None", "NotImplemented", "Ellipsis"].includes(value)) {
        html += token("constant", value);
      } else if (PYTHON_BUILTINS.has(value)) {
        html += token("builtin", value);
      } else {
        html += escapeHtml(value);
      }
      index += value.length;
      continue;
    }

    if (/^[+\-*/%=<>!&|^~:]+/.test(rest)) {
      const operator = rest.match(/^[+\-*/%=<>!&|^~:]+/)[0];
      html += token("operator", operator);
      index += operator.length;
      expectDefinition = false;
      continue;
    }

    html += escapeHtml(rest[0]);
    if (!/\s/.test(rest[0])) expectDefinition = false;
    index += 1;
  }

  return html;
}

function renderCodeBlock(code, infoString = "") {
  const language = String(infoString || "").trim().split(/\s+/)[0].toLowerCase();
  const normalizedCode = String(code || "").replace(/\n$/, "");
  const isPython = language === "python" || language === "py";
  const highlighted = isPython ? highlightPython(normalizedCode) : escapeHtml(normalizedCode);
  const label = language ? language.toUpperCase() : "CODE";
  const languageClass = language ? ` language-${escapeHtml(language)}` : "";
  return `<pre class="code-block" data-language="${escapeHtml(label)}"><button class="copy-code-button" type="button" data-copy-code aria-label="复制代码">复制</button><code class="${languageClass.trim()}">${highlighted}</code></pre>`;
}

async function copyCodeBlock(button) {
  const code = button.closest(".code-block")?.querySelector("code")?.textContent || "";
  if (!code) return;

  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(code);
  } else {
    const textarea = document.createElement("textarea");
    textarea.value = code;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("复制失败");
  }

  window.clearTimeout(button.copyResetTimer);
  button.textContent = "已复制";
  button.dataset.copied = "true";
  button.copyResetTimer = window.setTimeout(() => {
    button.textContent = "复制";
    delete button.dataset.copied;
  }, 1600);
}

function typesetMath(container = document.body) {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise([container]).catch((error) => console.error(error));
  }
}

function extractArticleToc(markdown) {
  const counts = new Map();
  return String(markdown || "")
    .replace(/\r\n/g, "\n")
    .replace(/^---\n[\s\S]*?\n---\n?/, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^#{1,4}\s+\S/.test(line))
    .map((line) => {
      const level = line.match(/^#+/)[0].length;
      const text = line
        .replace(/^#{1,4}\s+/, "")
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/\[\[([^\]]+)\]\]/g, "$1")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/\$([^$\n]+)\$/g, "$1")
        .trim();
      return { id: headingSlug(escapeHtml(text), counts), level, text };
    });
}

function getSearchTerm() {
  return els.search.value.trim().toLowerCase();
}

function getFilteredPosts(collection = state.posts, useFilter = true) {
  const term = getSearchTerm();
  const scopedPosts = [...collection].filter((post) => {
    return !useFilter || activeFilter === FILTER_ALL || post.category === activeFilter || post.tags.includes(activeFilter);
  });

  if (!term) {
    return scopedPosts.sort((a, b) => b.date.localeCompare(a.date));
  }

  const scoredPosts = scopedPosts
    .map((post) => ({ post, score: scorePost(post, term) }))
    .filter((item) => item.score > 0);
  const exactMatches = scoredPosts.filter((item) => item.score >= 100);
  const results = exactMatches.length ? exactMatches : scoredPosts;

  return results
    .sort((a, b) => b.score - a.score || b.post.date.localeCompare(a.post.date))
    .map((item) => item.post);
}

function scorePost(post, rawTerm) {
  const term = normalizeText(rawTerm);
  const title = normalizeText(post.title);
  const category = normalizeText(post.category);
  const tags = normalizeText(post.tags.join(" "));
      const content = normalizeText(`${post.excerpt || ""} ${post.content || ""} ${post.pdf || ""} ${post.pdfTitle || ""}`);
  const haystack = `${title} ${category} ${tags} ${content}`;

  let score = 0;
  if (title.includes(term)) score += 320;
  if (tags.includes(term)) score += 240;
  if (category.includes(term)) score += 180;
  if (content.includes(term)) score += 110;
  if (score > 0) return score;

  return Math.round(Math.max(similarity(term, title) * 95, similarity(term, tags) * 80, similarity(term, haystack) * 70));
}

function normalizeText(value) {
  return String(value || "").toLowerCase().replace(/\s+/g, "");
}

function similarity(query, text) {
  if (!query || !text) return 0;
  const queryChars = [...new Set(query)];
  const textSet = new Set(text);
  const overlap = queryChars.filter((char) => textSet.has(char)).length / queryChars.length;
  let ordered = 0;
  let start = 0;
  for (const char of query) {
    const index = text.indexOf(char, start);
    if (index >= 0) {
      ordered += 1;
      start = index + 1;
    }
  }
  return overlap * 0.7 + (ordered / query.length) * 0.3;
}

function renderProfile() {
  els.profileName.textContent = state.profile.name || "你的名字";
  els.profileBio.textContent = state.profile.bio || "请在 GitHub 中补充个人简介。";

  const slot = els.profileImage.closest(".photo-slot");
  if (state.profile.photo) {
    els.profileImage.src = state.profile.photo;
    slot.classList.add("has-image");
  } else {
    els.profileImage.removeAttribute("src");
    slot.classList.remove("has-image");
  }
}

function renderTaxonomy() {
  const categories = [...new Set(state.posts.map((post) => post.category).filter(Boolean))];
  const tags = [...new Set(state.posts.flatMap((post) => post.tags).filter(Boolean))];

  els.categoryCount.textContent = categories.length;
  els.tagCount.textContent = tags.length;
  els.categoryList.innerHTML = categories
    .map((category) => `<button type="button" data-filter="${escapeHtml(category)}">${escapeHtml(category)}</button>`)
    .join("");
  els.tagList.innerHTML = tags
    .map((tag) => `<button type="button" class="tag" data-filter="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`)
    .join("");

  const filters = [FILTER_ALL, ...categories, ...tags];
  els.blogFilters.innerHTML = filters
    .map(
      (filter) =>
        `<button type="button" class="${filter === activeFilter ? "active" : ""}" data-filter="${escapeHtml(filter)}">${escapeHtml(filter)}</button>`
    )
    .join("");
}

function createPostCard(post, type = "post", options = {}) {
  const template = document.querySelector("#postCardTemplate").content.cloneNode(true);
  const card = template.querySelector(".post-card");
  const meta = card.querySelector(".post-meta");
  if (options.pinned) {
    card.classList.add("is-pinned");
    const badge = document.createElement("span");
    badge.className = "pinned-badge";
    badge.textContent = "置顶";
    meta.append(badge);
  }
  meta.append(document.createTextNode(`${post.date} / ${post.category || "未分类"}`));
  card.querySelector("h3").textContent = post.title;
  card.querySelector("p").textContent = post.excerpt || excerpt(post.content);
  card.querySelector(".post-tags").innerHTML = post.tags
    .map((tag) => `<a href="#blog" data-filter="${escapeHtml(tag)}">${escapeHtml(tag)}</a>`)
    .join("");
  card.querySelector("button").addEventListener("click", () => openPost(post.id, type));
  card.querySelector("h3").addEventListener("click", () => openPost(post.id, type));
  return card;
}

function renderBlogArchive(posts) {
  if (!els.blogArchive) return;
  if (!posts.length) {
    els.blogArchive.innerHTML = '<div class="empty-state">当前筛选条件下没有可显示的文章归档。</div>';
    return;
  }

  const postsByYear = posts.reduce((groups, post) => {
    const year = String(post.date || "未定日期").slice(0, 4);
    if (!groups.has(year)) groups.set(year, []);
    groups.get(year).push(post);
    return groups;
  }, new Map());

  els.blogArchive.innerHTML = [...postsByYear.entries()]
    .map(([year, yearPosts]) => `
      <section class="archive-year">
        <header class="archive-year-head">
          <strong>${escapeHtml(year)}</strong>
          <span class="archive-year-dot" aria-hidden="true"></span>
          <span>${yearPosts.length} 篇帖子</span>
        </header>
        <div class="archive-entries">
          ${yearPosts
            .map((post) => {
              const monthDay = /^\d{4}-\d{2}-\d{2}$/.test(post.date || "") ? post.date.slice(5) : post.date || "待定";
              const label = post.tags[0] || post.category || "未分类";
              return `
                <a class="archive-entry" href="#post/${encodeURIComponent(post.id)}">
                  <time datetime="${escapeHtml(post.date || "")}">${escapeHtml(monthDay)}</time>
                  <span class="archive-track" aria-hidden="true"><i></i></span>
                  <strong>${escapeHtml(post.title)}</strong>
                  <span class="archive-entry-tag">#${escapeHtml(label)}</span>
                </a>
              `;
            })
            .join("")}
        </div>
      </section>
    `)
    .join("");
}

function renderPosts() {
  const posts = getFilteredPosts();
  const term = getSearchTerm();
  els.recentPosts.innerHTML = "";
  const postsByDate = [...state.posts].sort((a, b) => b.date.localeCompare(a.date));
  const pinnedPost = postsByDate.find((post) => post.id === HOME_PINNED_POST_ID);
  const homePosts = pinnedPost
    ? [pinnedPost, ...postsByDate.filter((post) => post.id !== HOME_PINNED_POST_ID)]
    : postsByDate;
  homePosts.slice(0, 4).forEach((post) => {
    els.recentPosts.append(createPostCard(post, "post", { pinned: post.id === HOME_PINNED_POST_ID }));
  });

  els.postList.innerHTML = "";
  renderBlogArchive(posts);
  if (!posts.length) {
    els.postList.innerHTML = '<div class="empty-state">没有匹配的文章。换个关键词试试，或者去 GitHub CMS 发布一篇新的。</div>';
    return;
  }

  if (term) {
    els.postList.innerHTML = `<div class="search-summary">搜索 “${escapeHtml(els.search.value.trim())}” ，按匹配度排序，共 ${posts.length} 篇。</div>`;
  }
  posts.forEach((post) => els.postList.append(createPostCard(post)));
}

function renderDiary() {
  if (!els.diaryList) return;
  const diary = getFilteredPosts(state.diary, false);
  els.diaryList.innerHTML = "";
  if (!diary.length) {
    els.diaryList.innerHTML = '<div class="empty-state">还没有日记。把 Markdown 文件放进 <code>content/diary</code> 后会显示在这里。</div>';
    return;
  }

  diary.forEach((entry) => els.diaryList.append(createPostCard(entry, "diary")));
}

function openPost(id, type = "post") {
  location.hash = `${type}/${encodeURIComponent(id)}`;
}

function getCollection(type) {
  return type === "diary" ? state.diary : state.posts;
}

function getListRoute(type) {
  return type === "diary" ? "diary" : "blog";
}

function renderPostPage(id, type = "post") {
  const collection = getCollection(type);
  const requestedId = decodeURIComponent(id || "");
  const resolvedId = type === "post" ? POST_REDIRECTS[requestedId] || requestedId : requestedId;
  const post = collection.find((item) => item.id === resolvedId);
  if (!post) {
    els.postReader.innerHTML = `<div class="empty-state">没有找到这篇内容。<a href="#${getListRoute(type)}">返回列表</a></div>`;
    return;
  }
  if (resolvedId !== requestedId) {
    history.replaceState(null, "", `#${type}/${encodeURIComponent(resolvedId)}`);
  }
  document.title = `${post.title} | Woman's World`;
  els.readerLayout?.classList.add("hide-left", "hide-right");

  const pdf = post.pdf
    ? `<p class="pdf-link"><strong>PDF：</strong><a href="${escapeHtml(post.pdf)}" target="_blank" rel="noreferrer">${escapeHtml(post.pdfTitle || "查看 PDF")}</a></p>`
    : "";

  els.postReader.innerHTML = `
    <div class="reader-actions">
      <a class="back-link" href="#${getListRoute(type)}">返回列表</a>
      <button class="secondary-button" type="button" data-reader-panel="left">收起书架</button>
      <button class="secondary-button" type="button" data-reader-panel="right">收起脉络</button>
    </div>
    <p class="eyebrow">${escapeHtml(post.category || "未分类")} / ${escapeHtml(post.date)}</p>
    <h1 class="reader-title">${escapeHtml(post.title)}</h1>
    ${pdf}
    ${markdownToHtml(post.content)}
    ${renderAdjacentPosts(collection, post, type)}
    <section class="comments-panel">
      <div class="section-heading">
        <h2>评论</h2>
        <span>GitHub 登录</span>
      </div>
      <div id="commentsMount"></div>
    </section>
  `;
  renderRelatedPosts(collection, post, type);
  renderArticleToc(post.content);
  syncReaderPanelButtons();
  typesetMath(els.postReader);
  renderComments(post, type);
}

function renderRelatedPosts(collection, currentPost, type) {
  if (!els.relatedPosts) return;
  const related = collection
    .filter((post) => post.category === currentPost.category)
    .sort((a, b) => b.date.localeCompare(a.date));

  const sidebarTitle = document.querySelector(".reader-sidebar .section-heading h2");
  if (sidebarTitle) sidebarTitle.textContent = currentPost.category || "未分类";

  els.relatedPosts.innerHTML =
    related
      .map(
        (post) => `
          <a class="reader-nav-item ${post.id === currentPost.id ? "active" : ""}" href="#${type}/${encodeURIComponent(post.id)}">
            <span>${escapeHtml(post.date)}</span>
            <strong>${escapeHtml(post.title)}</strong>
          </a>
        `
      )
      .join("") || '<div class="empty-state compact">这个目录下暂无其他内容。</div>';
}

function renderArticleToc(content) {
  if (!els.articleToc) return;
  const headings = extractArticleToc(content);
  els.articleToc.innerHTML =
    headings
      .map(
        (heading) => `
          <button class="article-toc-item level-${heading.level}" type="button" data-toc-target="${escapeHtml(heading.id)}">
            ${escapeHtml(heading.text)}
          </button>
        `
      )
      .join("") || '<div class="empty-state compact">这篇文章还没有标题结构。</div>';
}

function renderAdjacentPosts(collection, currentPost, type) {
  const sorted = [...collection].sort((a, b) => b.date.localeCompare(a.date));
  const index = sorted.findIndex((post) => post.id === currentPost.id);
  const previous = sorted[index - 1];
  const next = sorted[index + 1];

  if (!previous && !next) return "";

  return `
    <nav class="adjacent-posts" aria-label="上一篇和下一篇">
      ${
        previous
          ? `<a href="#${type}/${encodeURIComponent(previous.id)}"><span>上一篇</span><strong>${escapeHtml(previous.title)}</strong></a>`
          : "<span></span>"
      }
      ${
        next
          ? `<a href="#${type}/${encodeURIComponent(next.id)}"><span>下一篇</span><strong>${escapeHtml(next.title)}</strong></a>`
          : "<span></span>"
      }
    </nav>
  `;
}

function renderComments(post, type = "post") {
  const config = window.BLOG_COMMENTS || {};
  const mount = document.querySelector("#commentsMount");
  if (!mount || !config.enabled) return;
  mount.innerHTML = "";

  if (config.provider === "utterances") {
    const script = document.createElement("script");
    script.src = "https://utteranc.es/client.js";
    script.setAttribute("repo", config.repo);
    script.setAttribute("issue-term", `${config.issueTermPrefix || "entry"}:${type}:${post.id}`);
    script.setAttribute("label", config.label || "comment");
    script.setAttribute("theme", config.theme || "github-light");
    script.setAttribute("crossorigin", "anonymous");
    script.async = true;
    mount.append(script);
    return;
  }

  mount.innerHTML = '<div class="empty-state">评论系统尚未配置。</div>';
}

function renderResources() {
  const term = getSearchTerm();
  const collections = state.resources.filter((collection) => {
    const searchableText = [
      collection.title,
      collection.description,
      collection.content,
      ...(collection.items || []).flatMap((item) => [item.title, item.type, item.url, item.previewUrl, item.description])
    ]
      .join(" ")
      .toLowerCase();
    return searchableText.includes(term);
  });

  els.resourceList.innerHTML =
    collections
      .map(
        (collection, index) => {
          const items = collection.items || [];
          const files = items
            .map((item) => {
              const previewUrl = item.previewUrl || (item.type === "PDF" ? item.url : "");
              const itemTitle = item.title || "未命名资源";
              const actions = previewUrl
                ? `
                  <div class="resource-file-actions">
                    <button
                      class="resource-preview-button"
                      type="button"
                      data-resource-preview
                      data-preview-url="${escapeHtml(previewUrl)}"
                      data-download-url="${escapeHtml(item.url || previewUrl)}"
                      data-preview-title="${escapeHtml(itemTitle)}"
                      data-preview-meta="${escapeHtml(item.description || item.type || "资源文件")}"
                    >预览</button>
                    <a class="resource-download-link" href="${escapeHtml(item.url || previewUrl)}" download>下载</a>
                  </div>
                `
                : item.url
                  ? `<a class="resource-open-link" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">访问 <span aria-hidden="true">↗</span></a>`
                  : '<span class="resource-pending">待补充</span>';

              return `
                <article class="resource-file">
                  <span class="resource-file-type">${escapeHtml(item.type || "链接")}</span>
                  <div class="resource-file-copy">
                    <h4>${escapeHtml(itemTitle)}</h4>
                    <p>${escapeHtml(item.description || "暂无说明")}</p>
                  </div>
                  ${actions}
                </article>
              `;
            })
            .join("");

          return `
          <details class="resource-collection" id="resource-${escapeHtml(collection.slug || slugify(collection.title || `collection-${index + 1}`))}" ${term ? "open" : ""}>
            <summary class="resource-collection-summary">
              <div>
                <p class="resource-index">专题 ${String(index + 1).padStart(2, "0")}</p>
                <h3>${escapeHtml(collection.title || "未命名栏目")}</h3>
                <p>${escapeHtml(collection.description || "这里可以添加栏目简介。")}</p>
              </div>
              <div class="resource-summary-meta">
                <span class="resource-count">${items.length} 项资源</span>
                <span class="resource-expand-label">展开详情 <b aria-hidden="true">⌄</b></span>
              </div>
            </summary>
            <div class="resource-collection-body">
              <section class="resource-markdown reader">
                ${markdownToHtml(collection.content || "## 栏目导读\n\n在后台使用 Markdown 添加论文清单、学习路线或内容说明。")}
              </section>
              <section class="resource-files" aria-label="${escapeHtml(collection.title || "栏目")}的文件">
                <div class="resource-files-heading">
                  <h4>文件与链接</h4>
                  <span>PDF · PPT · CODE · WEB</span>
                </div>
                ${files || '<div class="empty-state compact">这个栏目还没有添加文件。</div>'}
              </section>
            </div>
          </details>
        `;
        }
      )
      .join("") || '<div class="empty-state">没有匹配的资源栏目。</div>';

  typesetMath(els.resourceList);
}

function resourcePreviewUrlAtZoom(url, zoom) {
  const baseUrl = String(url || "").split("#")[0];
  return `${baseUrl}#zoom=${zoom}`;
}

function applyResourcePreviewZoom(nextZoom = resourcePreviewZoom) {
  resourcePreviewZoom = Math.min(200, Math.max(50, Math.round(nextZoom / 10) * 10));
  if (els.resourcePreviewZoomReset) {
    els.resourcePreviewZoomReset.textContent = `${resourcePreviewZoom}%`;
  }
  if (els.resourcePreviewZoomOut) els.resourcePreviewZoomOut.disabled = resourcePreviewZoom <= 50;
  if (els.resourcePreviewZoomIn) els.resourcePreviewZoomIn.disabled = resourcePreviewZoom >= 200;

  if (resourcePreviewMode === "code" && els.resourcePreviewCode) {
    els.resourcePreviewCode.style.fontSize = `${0.94 * (resourcePreviewZoom / 100)}rem`;
  } else if (els.resourcePreviewFrame && resourcePreviewBaseUrl) {
    els.resourcePreviewFrame.src = resourcePreviewUrlAtZoom(resourcePreviewBaseUrl, resourcePreviewZoom);
  }
}

function resetResourcePreviewZoom() {
  resourcePreviewZoom = 100;
  if (els.resourcePreviewCode) els.resourcePreviewCode.style.removeProperty("font-size");
  applyResourcePreviewZoom(100);
}

function syncResourcePreviewFullscreenButton() {
  if (!els.resourcePreviewFullscreen) return;
  const isFullscreen = document.fullscreenElement === els.resourcePreviewShell
    || els.resourcePreviewDialog?.classList.contains("is-window-fullscreen");
  els.resourcePreviewFullscreen.textContent = isFullscreen ? "退出全屏" : "全屏";
  els.resourcePreviewFullscreen.setAttribute("aria-label", isFullscreen ? "退出全屏预览" : "全屏预览");
}

async function toggleResourcePreviewFullscreen() {
  if (!els.resourcePreviewDialog || !els.resourcePreviewShell) return;
  try {
    if (document.fullscreenElement === els.resourcePreviewShell) {
      await document.exitFullscreen();
    } else if (typeof els.resourcePreviewShell.requestFullscreen === "function") {
      await els.resourcePreviewShell.requestFullscreen();
    } else {
      els.resourcePreviewDialog.classList.toggle("is-window-fullscreen");
    }
  } catch {
    els.resourcePreviewDialog.classList.toggle("is-window-fullscreen");
  }
  syncResourcePreviewFullscreenButton();
}

function closeResourcePreview() {
  if (!els.resourcePreviewDialog) return;
  if (document.fullscreenElement === els.resourcePreviewShell) {
    document.exitFullscreen().catch(() => {});
  }
  els.resourcePreviewDialog.classList.remove("is-window-fullscreen");
  els.resourcePreviewDialog.close();
  els.resourcePreviewFrame.removeAttribute("src");
  els.resourcePreviewFrame.hidden = false;
  resourcePreviewBaseUrl = "";
  resourcePreviewMode = "frame";
  resetResourcePreviewZoom();
  syncResourcePreviewFullscreenButton();
  if (els.resourcePreviewCode) {
    els.resourcePreviewCode.hidden = true;
    els.resourcePreviewCode.textContent = "";
  }
}

function openImagePreview(button) {
  if (!els.imagePreviewDialog || !els.imagePreviewImage) return;
  const source = button.dataset.imageSrc;
  if (!source) return;
  const title = button.dataset.imageAlt || "图片预览";
  els.imagePreviewImage.src = source;
  els.imagePreviewImage.alt = title;
  if (els.imagePreviewTitle) els.imagePreviewTitle.textContent = title;
  if (!els.imagePreviewDialog.open) els.imagePreviewDialog.showModal();
}

function closeImagePreview() {
  if (!els.imagePreviewDialog?.open) return;
  els.imagePreviewDialog.close();
}

function openResourcePreview(button) {
  if (!els.resourcePreviewDialog) return;
  const previewUrl = button.dataset.previewUrl;
  const downloadUrl = button.dataset.downloadUrl || previewUrl;
  if (!previewUrl) return;

  els.resourcePreviewTitle.textContent = button.dataset.previewTitle || "资源预览";
  els.resourcePreviewMeta.textContent = button.dataset.previewMeta || "";
  resourcePreviewMode = "frame";
  resourcePreviewBaseUrl = previewUrl;
  els.resourcePreviewFrame.hidden = false;
  if (els.resourcePreviewCode) {
    els.resourcePreviewCode.hidden = true;
    els.resourcePreviewCode.textContent = "";
  }
  els.resourcePreviewDownload.href = downloadUrl;
  els.resourcePreviewDownload.setAttribute("download", "");
  els.resourcePreviewOpen.href = previewUrl;
  resetResourcePreviewZoom();
  els.resourcePreviewDialog.showModal();
}

async function openCodePreview(link) {
  if (!els.resourcePreviewDialog || !els.resourcePreviewCode) return;
  const previewUrl = link.getAttribute("href");
  if (!previewUrl) return;

  const fileName = link.textContent.trim() || previewUrl.split("/").pop() || "Python 程序";
  els.resourcePreviewTitle.textContent = fileName;
  els.resourcePreviewMeta.textContent = "LLM学习笔记2 · Python 配套程序";
  resourcePreviewMode = "code";
  resourcePreviewBaseUrl = previewUrl;
  els.resourcePreviewFrame.hidden = true;
  els.resourcePreviewFrame.removeAttribute("src");
  els.resourcePreviewCode.hidden = false;
  els.resourcePreviewCode.textContent = "正在加载代码…";
  els.resourcePreviewDownload.href = previewUrl;
  els.resourcePreviewDownload.setAttribute("download", fileName);
  els.resourcePreviewOpen.href = previewUrl;
  resetResourcePreviewZoom();
  if (!els.resourcePreviewDialog.open) els.resourcePreviewDialog.showModal();

  try {
    const response = await fetch(previewUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    els.resourcePreviewCode.innerHTML = highlightPython(await response.text());
  } catch {
    els.resourcePreviewCode.textContent = "代码预览加载失败，请使用上方按钮下载原文件。";
  }
}

function renderFriends() {
  if (!els.friendList) return;
  els.friendCount.textContent = state.friends.length;
  els.friendList.innerHTML =
    state.friends
      .map((friend) => {
        const title = friend.title || friend.name || "未命名友链";
        const subtitle = friend.subtitle || friend.description || friend.url || "";
        const avatar = friend.avatar || "resources/uploads/avatar.png";

        return `
          <a class="friend-link" href="${escapeHtml(friend.url || "#")}" target="_blank" rel="noreferrer">
            <img class="friend-avatar" src="${escapeHtml(avatar)}" alt="${escapeHtml(title)}" loading="lazy" />
            <span class="friend-copy">
              <strong>${escapeHtml(title)}</strong>
              <small>${escapeHtml(subtitle)}</small>
            </span>
          </a>
        `;
      })
      .join("") || '<div class="empty-state compact">还没有友链。</div>';
}

function planTypeMeta(plan, index) {
  const type = plan.type || ["checklist", "metric", "habit", "project"][index % 4];
  const meta = {
    checklist: { label: "清单", className: "is-checklist", icon: "✓", color: "var(--plan-check)" },
    metric: { label: "指标", className: "is-metric", icon: "%", color: "var(--plan-metric)" },
    habit: { label: "习惯", className: "is-habit", icon: "•", color: "var(--plan-habit)" },
    project: { label: "项目", className: "is-project", icon: "↗", color: "var(--plan-project)" }
  };
  return meta[type] || meta.checklist;
}

function planProgress(plan) {
  if (Number.isFinite(Number(plan.progress))) return Math.max(0, Math.min(100, Number(plan.progress)));
  if (Number.isFinite(Number(plan.current)) && Number.isFinite(Number(plan.target)) && Number(plan.target) > 0) {
    return Math.max(0, Math.min(100, Math.round((Number(plan.current) / Number(plan.target)) * 100)));
  }
  if (plan.status === "done") return 100;
  if (plan.status === "doing") return 56;
  return 12;
}

function renderPlans() {
  const done = state.plans.filter((plan) => plan.status === "done").length;
  const active = state.plans.filter((plan) => plan.status === "doing").length;
  const progress = state.plans.length ? Math.round((done / state.plans.length) * 100) : 0;
  if (els.progressValue) els.progressValue.textContent = `${progress}%`;
  document.querySelector(".progress-ring")?.style.setProperty("--progress", `${progress}%`);

  const heroTitle = document.querySelector("#planView .progress-panel h3");
  const heroCopy = document.querySelector("#planView .progress-panel p");
  if (heroTitle) heroTitle.textContent = `今天有 ${active} 个计划正在推进`;
  if (heroCopy) heroCopy.textContent = "清单、指标、习惯和长期项目会以不同形态展示；切换主题时，计划卡片会跟着当前色彩系统变化。";

  const statusLabel = {
    todo: "未开始",
    doing: "进行中",
    done: "已完成"
  };

  els.planList.innerHTML =
    state.plans
      .map((plan, index) => {
        const meta = planTypeMeta(plan, index);
        const itemProgress = planProgress(plan);
        const detail = plan.detail || plan.note || plan.description || "下一步还在酝酿中";
        return `
          <article class="plan-item ${meta.className}" style="--item-progress: ${itemProgress}%; --item-color: ${meta.color}">
            <div class="plan-item-main">
              <span class="plan-type-badge"><b>${meta.icon}</b>${meta.label}</span>
              <strong>${escapeHtml(plan.goal || plan.title || "未命名计划")}</strong>
              <p>${escapeHtml(detail)}</p>
            </div>
            <div class="plan-card-visual" aria-hidden="true">
              <span></span><span></span><span></span>
            </div>
            <div class="plan-item-footer">
              <span class="status">${statusLabel[plan.status] || "未开始"}</span>
              <span>${itemProgress}%</span>
            </div>
          </article>
        `;
      })
      .join("") || '<div class="empty-state">还没有计划。站主登录后可以新建第一个目标。</div>';
}
function renderPreview() {
  if (!els.postForm || !els.livePreview) return;
  const form = els.postForm;
  const title = form.elements.title.value || "文章标题预览";
  const content = form.elements.content.value || "# 文章标题预览\n\n支持 Obsidian 常用公式：\n\n$$\n(2,4), (3,6), (5,10)\n$$";
  els.livePreview.innerHTML = `<p class="eyebrow">实时预览</p>${markdownToHtml(`# ${title}\n\n${content}`)}`;
  typesetMath(els.livePreview);
}

function renderAll() {
  els.body.dataset.theme = state.theme;
  document.querySelectorAll("[data-theme-choice]").forEach((button) => {
    button.classList.toggle("active", button.dataset.themeChoice === state.theme);
  });
  renderProfile();
  renderTaxonomy();
  renderPosts();
  renderDiary();
  renderResources();
  renderFriends();
  renderPlans();
  renderPreview();
  updateWriterAuthView();
}

function setRoute(rawRoute) {
  const [route, id] = (rawRoute || "home").split("/");
  const view = route === "diary" && id ? "post" : routeMap[route] ? route : "home";
  els.body.dataset.route = view;

  Object.entries(routeMap).forEach(([key, element]) => {
    element.classList.toggle("active", key === view);
  });
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("active", link.dataset.route === (view === "post" ? route : view));
  });

  if (view === "post") {
    renderPostPage(route === "diary" ? id : id, route === "diary" ? "diary" : "post");
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else {
    document.title = "Woman's World";
  }
}

function buildDraftMarkdown() {
  const data = new FormData(els.postForm);
  const title = data.get("title").trim() || "未命名文章";
  const category = data.get("category").trim() || "未分类";
  const tags = data
    .get("tags")
    .split(/[,，]/)
    .map((tag) => tag.trim())
    .filter(Boolean);
  const pdf = data.get("pdf").trim();
  const pdfTitle = data.get("pdfTitle")?.trim() || "";
  const content = data.get("content").trim() || `# ${title}\n\n正文待补充。`;
  return `---\ntitle: "${title.replaceAll('"', '\\"')}"\ncategory: "${category.replaceAll('"', '\\"')}"\ntags:\n${tags.map((tag) => `  - "${tag.replaceAll('"', '\\"')}"`).join("\n") || "  []"}\ndate: "${new Date().toISOString().slice(0, 10)}"\npdf: "${pdf.replaceAll('"', '\\"')}"\npdfTitle: "${pdfTitle.replaceAll('"', '\\"')}"\n---\n\n${content}\n`;
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
}

function textToBase64(text) {
  return bytesToBase64(new TextEncoder().encode(text));
}

function base64ToText(base64) {
  const binary = atob(base64.replace(/\s/g, ""));
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

function escapeYaml(value) {
  return String(value || "").replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

function parseMarkdownFile(source, fallbackTitle = "") {
  const match = String(source || "").match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) {
    return {
      meta: {
        title: fallbackTitle,
        category: "",
        tags: [],
        date: new Date().toISOString().slice(0, 10),
        pdf: "",
        pdfTitle: "",
        summary: ""
      },
      content: String(source || "")
    };
  }

  const meta = {};
  let currentKey = "";
  match[1].split("\n").forEach((line) => {
    const listItem = line.match(/^\s*-\s+["']?(.+?)["']?\s*$/);
    if (listItem && currentKey) {
      meta[currentKey] = Array.isArray(meta[currentKey]) ? meta[currentKey] : [];
      meta[currentKey].push(listItem[1]);
      return;
    }

    const pair = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!pair) return;
    currentKey = pair[1];
    const value = pair[2].trim();
    meta[currentKey] = value ? value.replace(/^["']|["']$/g, "") : [];
  });

  return {
    meta: {
      title: meta.title || fallbackTitle,
      category: Array.isArray(meta.category) ? meta.category[0] || "" : meta.category || "",
      tags: Array.isArray(meta.tags) ? meta.tags : [],
      date: meta.date || new Date().toISOString().slice(0, 10),
      pdf: meta.pdf || "",
      pdfTitle: meta.pdfTitle || "",
      summary: Array.isArray(meta.summary) ? meta.summary[0] || "" : meta.summary || ""
    },
    content: match[2].trim()
  };
}

function buildWriterMarkdown() {
  const title = els.writerTitle.value.trim() || "未命名文章";
  const category = els.writerCategory.value.trim() || "未分类";
  const tags = els.writerTags.value
    .split(/[,，]/)
    .map((tag) => tag.trim())
    .filter(Boolean);
  const date = els.writerDate.value || new Date().toISOString().slice(0, 10);
  const pdf = els.writerPdf.value.trim();
  const pdfTitle = els.writerPdfTitle.value.trim();
  const summary = els.writerSummary.value.trim().replace(/\s+/g, " ");
  const content = els.writerContent.value.trim() || `# ${title}\n\n正文待补充。`;

  return `---\ntitle: "${escapeYaml(title)}"\ncategory: "${escapeYaml(category)}"\ntags:\n${tags.map((tag) => `  - "${escapeYaml(tag)}"`).join("\n") || "  []"}\ndate: "${escapeYaml(date)}"\nsummary: "${escapeYaml(summary)}"\npdf: "${escapeYaml(pdf)}"\npdfTitle: "${escapeYaml(pdfTitle)}"\n---\n\n${content}\n`;
}

function setWriterMessage(target, message, type = "info") {
  if (!target) return;
  target.textContent = message;
  target.dataset.type = type;
}

async function githubRequest(path, options = {}) {
  if (!writerState.token) throw new Error("请先登录写作台。");
  const response = await fetch(path.startsWith("https://") ? path : `https://api.github.com${path}`, {
    ...options,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${writerState.token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {})
    }
  });

  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).message || "";
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || `GitHub API ${response.status}`);
  }

  if (response.status === 204) return null;
  return response.json();
}

function updateWriterAuthView() {
  const isOwner = writerState.user?.login === GITHUB_OWNER;
  if (els.writerNav) els.writerNav.hidden = !isOwner;
  if (els.planOwnerPanel) els.planOwnerPanel.hidden = !isOwner;
  if (els.writerLoginPanel) els.writerLoginPanel.hidden = Boolean(isOwner);
  if (els.writerStudio) els.writerStudio.hidden = !isOwner;
  if (els.writerAuthStatus) {
    els.writerAuthStatus.textContent = isOwner
      ? `已以 ${writerState.user.login} 身份连接 GitHub，保存会提交到 ${GITHUB_OWNER}/${GITHUB_REPO}。`
      : "";
  }
}

function resetWriterForm() {
  writerState.currentPath = "";
  writerState.currentSha = "";
  if (els.writerPostSelect) els.writerPostSelect.value = "";
  if (els.writerCurrentPath) els.writerCurrentPath.textContent = "新建文章";
  if (els.writerTitle) els.writerTitle.value = "";
  if (els.writerDate) els.writerDate.value = new Date().toISOString().slice(0, 10);
  if (els.writerCategory) els.writerCategory.value = "";
  if (els.writerTags) els.writerTags.value = "";
  if (els.writerPdf) els.writerPdf.value = "";
  if (els.writerPdfTitle) els.writerPdfTitle.value = "";
  if (els.writerSummary) els.writerSummary.value = "";
  if (els.writerContent) els.writerContent.value = "";
  renderWriterPreview();
}

function fillWriterForm(markdown, file) {
  const parsed = parseMarkdownFile(markdown, file?.name?.replace(/\.md$/i, "") || "");
  writerState.currentPath = file?.path || "";
  writerState.currentSha = file?.sha || "";
  els.writerTitle.value = parsed.meta.title;
  els.writerDate.value = parsed.meta.date;
  els.writerCategory.value = parsed.meta.category;
  els.writerTags.value = parsed.meta.tags.join(", ");
  els.writerPdf.value = parsed.meta.pdf;
  els.writerPdfTitle.value = parsed.meta.pdfTitle;
  els.writerSummary.value = parsed.meta.summary;
  els.writerContent.value = parsed.content;
  els.writerCurrentPath.textContent = writerState.currentPath || "新建文章";
  renderWriterPreview();
}

function setWriterPreviewStatus(message, state = "synced") {
  if (!els.writerPreviewStatus) return;
  els.writerPreviewStatus.textContent = message;
  els.writerPreviewStatus.dataset.state = state;
}

function updateWriterSummaryPreview() {
  const summary = els.writerSummary?.value?.trim() || "";
  const content = els.writerContent?.value || "";
  const summaryFallback = excerpt(content, 180) || "摘要会显示在这里。";
  if (els.writerSummaryCount) els.writerSummaryCount.textContent = `${summary.length} / 180`;
  if (els.writerSummaryPreview) els.writerSummaryPreview.textContent = summary || summaryFallback;
}

function cancelWriterPreviewSchedule() {
  if (writerPreviewTimer) window.clearTimeout(writerPreviewTimer);
  writerPreviewTimer = null;
  if (writerPreviewIdleHandle !== null && "cancelIdleCallback" in window) {
    window.cancelIdleCallback(writerPreviewIdleHandle);
  }
  writerPreviewIdleHandle = null;
  if (writerMathIdleHandle !== null && "cancelIdleCallback" in window) {
    window.cancelIdleCallback(writerMathIdleHandle);
  }
  writerMathIdleHandle = null;
}

function scheduleWriterMath(revision, immediate = false) {
  const content = els.writerContent?.value || "";
  if (!content.includes("$") && !content.includes("\\(")) return;
  const run = () => {
    writerMathIdleHandle = null;
    if (revision !== writerPreviewRevision) return;
    typesetMath(els.writerPreview);
  };
  if (immediate || !("requestIdleCallback" in window)) {
    window.setTimeout(run, immediate ? 0 : 700);
  } else {
    writerMathIdleHandle = window.requestIdleCallback(run, { timeout: 2200 });
  }
}

function scheduleWriterPreview({ immediate = false } = {}) {
  if (!els.writerPreview) return;
  updateWriterSummaryPreview();
  cancelWriterPreviewSchedule();
  const revision = ++writerPreviewRevision;
  const length = els.writerContent?.value?.length || 0;
  const delay = immediate ? 0 : length > 60000 ? 1100 : length > 20000 ? 700 : 380;
  setWriterPreviewStatus(immediate ? "正在更新…" : "等待输入停止…", "pending");

  const render = () => {
    writerPreviewIdleHandle = null;
    if (revision !== writerPreviewRevision) return;
    renderWriterPreview({ revision, typesetImmediately: immediate });
  };
  writerPreviewTimer = window.setTimeout(() => {
    writerPreviewTimer = null;
    if ("requestIdleCallback" in window && !immediate) {
      writerPreviewIdleHandle = window.requestIdleCallback(render, { timeout: 1200 });
    } else {
      render();
    }
  }, delay);
}

function renderWriterPreview({ revision = ++writerPreviewRevision, typesetImmediately = true } = {}) {
  if (!els.writerPreview) return;
  cancelWriterPreviewSchedule();
  const title = els.writerTitle?.value?.trim() || "文章标题";
  const date = els.writerDate?.value || new Date().toISOString().slice(0, 10);
  const category = els.writerCategory?.value?.trim() || "未分类";
  const pdf = els.writerPdf?.value?.trim();
  const pdfTitle = els.writerPdfTitle?.value?.trim();
  const content = els.writerContent?.value || "";
  updateWriterSummaryPreview();
  setWriterPreviewStatus("正在更新…", "pending");
  if (window.MathJax?.typesetClear) {
    window.MathJax.typesetClear([els.writerPreview]);
  }
  els.writerPreview.innerHTML = `
    <p class="eyebrow">${escapeHtml(category)} / ${escapeHtml(date)}</p>
    <h1 class="reader-title">${escapeHtml(title)}</h1>
    ${pdf ? `<p class="pdf-link"><strong>PDF：</strong><a href="${escapeHtml(pdf)}" target="_blank" rel="noreferrer">${escapeHtml(pdfTitle || "查看 PDF")}</a></p>` : ""}
    ${markdownToHtml(content)}
  `;
  setWriterPreviewStatus(`已同步 · ${content.length.toLocaleString()} 字符`, "synced");
  scheduleWriterMath(revision, typesetImmediately);
}

async function verifyWriterToken(token, silent = false) {
  writerState.token = token.trim();
  if (!writerState.token) throw new Error("请先填写 GitHub Token。");
  const user = await githubRequest("/user");
  if (user.login !== GITHUB_OWNER) {
    writerState.token = "";
    sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    throw new Error(`当前登录用户是 ${user.login}，不是站长 ${GITHUB_OWNER}。`);
  }
  writerState.user = user;
  sessionStorage.setItem(TOKEN_STORAGE_KEY, writerState.token);
  updateWriterAuthView();
  if (!silent) setWriterMessage(els.writerLoginMessage, "登录成功，写作台已打开。", "success");
  await loadWriterPosts();
  resetWriterForm();
}

async function loadWriterPosts() {
  if (!els.writerPostSelect || writerState.user?.login !== GITHUB_OWNER) return;
  const files = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/content/posts?ref=${GITHUB_BRANCH}`);
  const markdownFiles = files
    .filter((file) => file.type === "file" && file.name.endsWith(".md"))
    .sort((a, b) => a.name.localeCompare(b.name));

  writerState.files = await Promise.all(markdownFiles.map(async (file) => {
    const fallbackTitle = file.name.replace(/\.md$/i, "");
    try {
      const detail = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encodeURIComponent(file.path).replaceAll("%2F", "/")}?ref=${GITHUB_BRANCH}`);
      const markdown = base64ToText(detail.content || "");
      const title = parseMarkdownFile(markdown, fallbackTitle).meta.title.trim() || fallbackTitle;
      return { ...file, ...detail, markdown, displayTitle: title };
    } catch {
      return { ...file, displayTitle: fallbackTitle };
    }
  }));

  els.writerPostSelect.innerHTML = '<option value="">新建文章</option>';
  writerState.files.forEach((file) => {
    const option = document.createElement("option");
    option.value = file.path;
    option.textContent = file.displayTitle;
    els.writerPostSelect.append(option);
  });
}

async function loadWriterPost(path) {
  if (!path) {
    resetWriterForm();
    return;
  }
  const cachedFile = writerState.files.find((file) => file.path === path);
  if (cachedFile?.markdown) {
    fillWriterForm(cachedFile.markdown, cachedFile);
    return;
  }
  const file = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encodeURIComponent(path).replaceAll("%2F", "/")}?ref=${GITHUB_BRANCH}`);
  fillWriterForm(base64ToText(file.content), file);
}

async function saveGithubFile(path, contentBase64, message, sha = "") {
  const body = {
    message,
    content: contentBase64,
    branch: GITHUB_BRANCH
  };
  if (sha) body.sha = sha;
  return githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encodeURIComponent(path).replaceAll("%2F", "/")}`, {
    method: "PUT",
    body: JSON.stringify(body)
  });
}

async function saveWriterPost() {
  const title = els.writerTitle.value.trim();
  if (!title) throw new Error("请先填写文章标题，再保存到 GitHub。");
  const path = writerState.currentPath || `content/posts/${slugify(title)}.md`;
  let sha = writerState.currentSha;

  if (!sha) {
    try {
      const existing = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encodeURIComponent(path).replaceAll("%2F", "/")}?ref=${GITHUB_BRANCH}`);
      sha = existing.sha;
    } catch {
      sha = "";
    }
  }

  const result = await saveGithubFile(path, textToBase64(buildWriterMarkdown()), `Update blog post: ${title}`, sha);
  writerState.currentPath = path;
  writerState.currentSha = result.content.sha;
  els.writerCurrentPath.textContent = path;
  await loadWriterPosts();
  els.writerPostSelect.value = path;
  setWriterMessage(els.writerSaveMessage, "已提交到 GitHub。GitHub Actions 部署完成后，线上博客会自动更新。", "success");
}

function insertAtTextarea(textarea, text) {
  const start = textarea.selectionStart ?? textarea.value.length;
  const end = textarea.selectionEnd ?? textarea.value.length;
  textarea.value = `${textarea.value.slice(0, start)}${text}${textarea.value.slice(end)}`;
  textarea.selectionStart = textarea.selectionEnd = start + text.length;
  textarea.focus();
}

function writerImageExtension(file) {
  const nameExtension = String(file?.name || "").split(".").pop().toLowerCase();
  if (/^(png|jpe?g|gif|webp|svg|bmp|avif)$/.test(nameExtension)) {
    return nameExtension.replace("jpeg", "jpg");
  }
  return (String(file?.type || "").split("/")[1] || "png").replace("jpeg", "jpg").replace("svg+xml", "svg");
}

async function uploadWriterImageBatch(files, prefix = "pasted") {
  const images = [...files];
  const stamp = Date.now();
  const prepared = [];
  for (const [index, file] of images.entries()) {
    setWriterMessage(els.writerSaveMessage, `正在准备图片 ${index + 1} / ${images.length}：${file.name}`, "info");
    const extension = writerImageExtension(file);
    const path = `resources/uploads/${prefix}-${stamp}-${index + 1}.${extension}`;
    const bytes = new Uint8Array(await file.arrayBuffer());
    prepared.push({ file, path, content: bytesToBase64(bytes) });
  }

  const ref = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/git/ref/heads/${GITHUB_BRANCH}`);
  const parentCommit = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/git/commits/${ref.object.sha}`);
  const treeEntries = [];
  for (const [index, item] of prepared.entries()) {
    setWriterMessage(els.writerSaveMessage, `正在上传图片 ${index + 1} / ${prepared.length}：${item.file.name}`, "info");
    const blob = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/git/blobs`, {
      method: "POST",
      body: JSON.stringify({ content: item.content, encoding: "base64" })
    });
    treeEntries.push({ path: item.path, mode: "100644", type: "blob", sha: blob.sha });
  }

  const tree = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/git/trees`, {
    method: "POST",
    body: JSON.stringify({ base_tree: parentCommit.tree.sha, tree: treeEntries })
  });
  const commit = await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/git/commits`, {
    method: "POST",
    body: JSON.stringify({
      message: `Upload ${prepared.length} ${prefix === "obsidian" ? "Obsidian" : "pasted"} image${prepared.length > 1 ? "s" : ""}`,
      tree: tree.sha,
      parents: [ref.object.sha]
    })
  });
  await githubRequest(`/repos/${GITHUB_OWNER}/${GITHUB_REPO}/git/refs/heads/${GITHUB_BRANCH}`, {
    method: "PATCH",
    body: JSON.stringify({ sha: commit.sha, force: false })
  });

  return prepared.map((item) => ({ file: item.file, url: `/my-blog/${item.path}` }));
}

function imageReferenceBaseName(reference) {
  let normalized = String(reference || "").trim().replace(/^<|>$/g, "");
  normalized = normalized.split(/[?#]/)[0];
  try {
    normalized = decodeURIComponent(normalized);
  } catch {
    // 保留无法解码的原始文件名，后续仍可尝试匹配。
  }
  return normalized.split(/[\\/]/).pop().toLowerCase();
}

function isLocalImageReference(reference) {
  const value = String(reference || "").trim().replace(/^<|>$/g, "");
  return !/^(?:https?:|data:|blob:)/i.test(value)
    && !/^(?:\.\/)?resources\//i.test(value)
    && !value.startsWith("/my-blog/")
    && !value.startsWith("/resources/");
}

function hasObsidianImageReferences(markdown) {
  return localImageReferenceNames(markdown).size > 0;
}

function localImageReferenceNames(markdown) {
  const text = String(markdown || "");
  const names = new Set();
  for (const match of text.matchAll(/!\[\[([^\]\n]+\.(?:png|jpe?g|gif|webp|svg|bmp|avif))(?:\|[^\]]+)?\]\]/gi)) {
    names.add(imageReferenceBaseName(match[1]));
  }
  for (const match of text.matchAll(/!\[[^\]]*\]\(([^)\n]+)\)/gi)) {
    const name = imageReferenceBaseName(match[1]);
    if (isLocalImageReference(match[1]) && /\.(?:png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(name)) {
      names.add(name);
    }
  }
  return names;
}

function replaceLocalImageReferences(markdown, fileName, uploadedUrl) {
  const targetName = imageReferenceBaseName(fileName);
  let replacements = 0;
  let output = String(markdown || "").replace(
    /!\[\[([^\]\n]+\.(?:png|jpe?g|gif|webp|svg|bmp|avif))(?:\|([^\]]+))?\]\]/gi,
    (match, source, option) => {
      if (imageReferenceBaseName(source) !== targetName) return match;
      const optionText = String(option || "").trim();
      const alt = optionText && !/^\d+(?:x\d+)?$/i.test(optionText)
        ? optionText
        : String(fileName).replace(/\.[^.]+$/, "");
      replacements += 1;
      return `![${alt}](${uploadedUrl})`;
    }
  );

  output = output.replace(/!\[([^\]]*)\]\(([^)\n]+)\)/g, (match, alt, source) => {
    if (!isLocalImageReference(source) || imageReferenceBaseName(source) !== targetName) return match;
    replacements += 1;
    return `![${alt || String(fileName).replace(/\.[^.]+$/, "")}](${uploadedUrl})`;
  });

  return { markdown: output, replacements };
}

async function importWriterImages(files, prefix = "obsidian") {
  if (writerImageUploadInProgress) throw new Error("上一批图片仍在上传，请稍候再试。");
  writerImageUploadInProgress = true;
  try {
    return await performWriterImageImport(files, prefix);
  } finally {
    writerImageUploadInProgress = false;
  }
}

async function performWriterImageImport(files, prefix = "obsidian") {
  const images = [...(files || [])].filter((file) => file?.type?.startsWith("image/") || /\.(?:png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(file?.name || ""));
  if (!images.length) throw new Error("没有选择可上传的图片文件。");

  let markdown = els.writerContent.value;
  let replacedCount = 0;
  const unmatchedLinks = [];
  const uploaded = await uploadWriterImageBatch(images, prefix);
  for (const { file, url } of uploaded) {
    const result = replaceLocalImageReferences(markdown, file.name, url);
    markdown = result.markdown;
    replacedCount += result.replacements;
    if (!result.replacements) {
      unmatchedLinks.push(`![${file.name.replace(/\.[^.]+$/, "")}](${url})`);
    }
  }

  els.writerContent.value = markdown;
  if (unmatchedLinks.length) {
    insertAtTextarea(els.writerContent, `\n${unmatchedLinks.join("\n")}\n`);
  }
  renderWriterPreview();
  const remaining = hasObsidianImageReferences(els.writerContent.value);
  const message = remaining
    ? `已上传 ${images.length} 张图片并替换 ${replacedCount} 处链接；仍有未匹配附件，请继续选择对应文件。`
    : `已上传 ${images.length} 张图片并替换 ${replacedCount} 处链接，预览已更新。`;
  setWriterMessage(els.writerSaveMessage, message, remaining ? "info" : "success");
}

function setObsidianDirectoryStatus(message, connected = false) {
  if (els.writerObsidianStatus) {
    els.writerObsidianStatus.innerHTML = message;
    els.writerObsidianStatus.dataset.connected = connected ? "true" : "false";
  }
  if (els.writerObsidianImagesButton) {
    els.writerObsidianImagesButton.textContent = connected ? "更换 Obsidian 图片文件夹" : "关联 Obsidian 图片文件夹";
  }
}

function openWriterSettingsDb() {
  if (!("indexedDB" in window)) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(WRITER_SETTINGS_DB, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains("settings")) {
        request.result.createObjectStore("settings");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveObsidianDirectoryHandle(handle) {
  try {
    const db = await openWriterSettingsDb();
    if (!db) return;
    await new Promise((resolve, reject) => {
      const transaction = db.transaction("settings", "readwrite");
      transaction.objectStore("settings").put(handle, OBSIDIAN_DIRECTORY_KEY);
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
    });
    db.close();
  } catch {
    // 某些隐私模式不允许持久化文件夹句柄，当前页面仍可继续使用。
  }
}

async function loadObsidianDirectoryHandle() {
  try {
    const db = await openWriterSettingsDb();
    if (!db) return null;
    const handle = await new Promise((resolve, reject) => {
      const request = db.transaction("settings", "readonly").objectStore("settings").get(OBSIDIAN_DIRECTORY_KEY);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
    db.close();
    return handle;
  } catch {
    return null;
  }
}

async function hasObsidianDirectoryPermission(handle, request = false) {
  if (!handle) return false;
  const options = { mode: "read" };
  if (typeof handle.queryPermission === "function" && await handle.queryPermission(options) === "granted") {
    return true;
  }
  return Boolean(request
    && typeof handle.requestPermission === "function"
    && await handle.requestPermission(options) === "granted");
}

async function restoreObsidianDirectory() {
  const handle = await loadObsidianDirectoryHandle();
  if (!handle) return;
  obsidianDirectoryHandle = handle;
  if (await hasObsidianDirectoryPermission(handle)) {
    setObsidianDirectoryStatus(`已关联：<strong>${escapeHtml(handle.name)}</strong>。粘贴 Obsidian 正文时会自动同步图片。`, true);
  } else {
    setObsidianDirectoryStatus(`已记住文件夹 <strong>${escapeHtml(handle.name)}</strong>，点击按钮恢复读取权限。`);
  }
}

async function buildObsidianImageIndex(directoryHandle) {
  const index = new Map();
  async function visit(handle) {
    for await (const entry of handle.values()) {
      if (entry.kind === "directory") {
        await visit(entry);
      } else if (/\.(?:png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(entry.name)) {
        const key = entry.name.toLowerCase();
        if (!index.has(key)) index.set(key, []);
        index.get(key).push(entry);
      }
    }
  }
  await visit(directoryHandle);
  return index;
}

async function syncObsidianImagesFromDirectory() {
  const referenceNames = localImageReferenceNames(els.writerContent?.value || "");
  if (!referenceNames.size) {
    setWriterMessage(els.writerSaveMessage, "正文中没有需要同步的 Obsidian 本地图片链接。", "info");
    return;
  }
  if (!await hasObsidianDirectoryPermission(obsidianDirectoryHandle)) {
    setWriterMessage(els.writerSaveMessage, "请先点击“关联 Obsidian 图片文件夹”恢复读取权限。", "info");
    return;
  }

  setWriterMessage(els.writerSaveMessage, `正在附件文件夹中查找 ${referenceNames.size} 张图片...`, "info");
  obsidianImageIndex = await buildObsidianImageIndex(obsidianDirectoryHandle);
  const files = [];
  const missing = [];
  const duplicates = [];
  for (const name of referenceNames) {
    const matches = obsidianImageIndex.get(name) || [];
    if (!matches.length) {
      missing.push(name);
      continue;
    }
    if (matches.length > 1) duplicates.push(name);
    files.push(await matches[0].getFile());
  }

  if (files.length) await importWriterImages(files, "obsidian");
  const notes = [];
  if (missing.length) notes.push(`未找到：${missing.join("、")}`);
  if (duplicates.length) notes.push(`发现同名文件并使用第一个：${duplicates.join("、")}`);
  if (notes.length) {
    setWriterMessage(els.writerSaveMessage, `已完成可匹配图片的同步。${notes.join("；")}。`, missing.length ? "info" : "success");
  }
}

async function connectObsidianDirectory() {
  if (obsidianDirectoryHandle && await hasObsidianDirectoryPermission(obsidianDirectoryHandle, true)) {
    obsidianImageIndex = null;
    setObsidianDirectoryStatus(`已关联：<strong>${escapeHtml(obsidianDirectoryHandle.name)}</strong>。粘贴 Obsidian 正文时会自动同步图片。`, true);
    await syncObsidianImagesFromDirectory();
    return;
  }
  if (typeof window.showDirectoryPicker !== "function") {
    els.writerObsidianImagesInput.click();
    return;
  }
  const handle = await window.showDirectoryPicker({ id: "obsidian-attachments", mode: "read" });
  obsidianDirectoryHandle = handle;
  obsidianImageIndex = null;
  await saveObsidianDirectoryHandle(handle);
  setObsidianDirectoryStatus(`已关联：<strong>${escapeHtml(handle.name)}</strong>。粘贴 Obsidian 正文时会自动同步图片。`, true);
  await syncObsidianImagesFromDirectory();
}

async function importObsidianFolderFallback(fileList) {
  const referenceNames = localImageReferenceNames(els.writerContent?.value || "");
  const files = [...(fileList || [])].filter((file) => referenceNames.has(file.name.toLowerCase()));
  if (!files.length) throw new Error("所选文件夹中没有找到与正文链接同名的图片。");
  await importWriterImages(files, "obsidian");
  setObsidianDirectoryStatus("当前浏览器无法长期关联文件夹；本次已从所选目录批量导入。", true);
}

function downloadText(fileName, text) {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

function syncReaderPanelButtons() {
  if (!els.readerLayout) return;
  document.querySelectorAll("[data-reader-panel='left']").forEach((button) => {
    button.textContent = els.readerLayout.classList.contains("hide-left") ? "展开书架" : "收起书架";
  });
  document.querySelectorAll("[data-reader-panel='right']").forEach((button) => {
    button.textContent = els.readerLayout.classList.contains("hide-right") ? "展开脉络" : "收起脉络";
  });
}

document.addEventListener("click", async (event) => {
  const imagePreviewButton = event.target.closest("[data-image-preview]");
  if (imagePreviewButton) {
    openImagePreview(imagePreviewButton);
    return;
  }

  const codeResourceLink = event.target.closest('.reader a[href$=".py"]');
  if (codeResourceLink) {
    event.preventDefault();
    await openCodePreview(codeResourceLink);
    return;
  }

  const copyButton = event.target.closest("[data-copy-code]");
  if (copyButton) {
    try {
      await copyCodeBlock(copyButton);
    } catch {
      copyButton.textContent = "复制失败";
      copyButton.dataset.copied = "true";
    }
    return;
  }

  const readerPanelButton = event.target.closest("[data-reader-panel]");
  if (readerPanelButton && els.readerLayout) {
    const side = readerPanelButton.dataset.readerPanel;
    els.readerLayout.classList.toggle(side === "left" ? "hide-left" : "hide-right");
    syncReaderPanelButtons();
    return;
  }

  const tocButton = event.target.closest("[data-toc-target]");
  if (tocButton) {
    const target = document.getElementById(tocButton.dataset.tocTarget);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      document.querySelectorAll(".article-toc-item").forEach((item) => item.classList.toggle("active", item === tocButton));
    }
    return;
  }

  const themeButton = event.target.closest("[data-theme-choice]");
  if (themeButton) {
    state.theme = themeButton.dataset.themeChoice;
    renderAll();
  }

  const filterButton = event.target.closest("[data-filter]");
  if (filterButton) {
    activeFilter = filterButton.dataset.filter;
    location.hash = "blog";
    renderTaxonomy();
    renderPosts();
  }
});

document.addEventListener("pointermove", (event) => {
  if (event.pointerType === "touch") return;
  document.querySelectorAll(".post-card").forEach((card) => {
    const rect = card.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const distance = Math.hypot(event.clientX - centerX, event.clientY - centerY);
    const influence = Math.max(0, 1 - distance / 340);

    if (influence <= 0) {
      card.classList.remove("is-near");
      card.style.removeProperty("--card-lift");
      card.style.removeProperty("--card-scale");
      card.style.removeProperty("--card-shadow-alpha");
      return;
    }

    card.classList.add("is-near");
    card.style.setProperty("--card-lift", `${(-4 - influence * 14).toFixed(2)}px`);
    card.style.setProperty("--card-scale", (1 + influence * 0.055).toFixed(3));
    card.style.setProperty("--card-shadow-alpha", (0.12 + influence * 0.18).toFixed(3));
  });
});

document.addEventListener("pointerleave", () => {
  document.querySelectorAll(".post-card.is-near").forEach((card) => {
    card.classList.remove("is-near");
    card.style.removeProperty("--card-lift");
    card.style.removeProperty("--card-scale");
    card.style.removeProperty("--card-shadow-alpha");
  });
});

window.addEventListener("hashchange", () => setRoute(location.hash.replace("#", "")));

els.search.addEventListener("input", () => {
  renderPosts();
  renderDiary();
  renderResources();
});

els.search.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  activeFilter = FILTER_ALL;
  location.hash = "blog";
  renderTaxonomy();
  renderPosts();
});

if (els.resourceList) {
  els.resourceList.addEventListener("click", (event) => {
    const previewButton = event.target.closest("[data-resource-preview]");
    if (previewButton) openResourcePreview(previewButton);
  });
}

if (els.resourcePreviewClose) {
  els.resourcePreviewClose.addEventListener("click", closeResourcePreview);
}

if (els.resourcePreviewZoomOut) {
  els.resourcePreviewZoomOut.addEventListener("click", () => applyResourcePreviewZoom(resourcePreviewZoom - 10));
}

if (els.resourcePreviewZoomReset) {
  els.resourcePreviewZoomReset.addEventListener("click", resetResourcePreviewZoom);
}

if (els.resourcePreviewZoomIn) {
  els.resourcePreviewZoomIn.addEventListener("click", () => applyResourcePreviewZoom(resourcePreviewZoom + 10));
}

if (els.resourcePreviewFullscreen) {
  els.resourcePreviewFullscreen.addEventListener("click", toggleResourcePreviewFullscreen);
}

document.addEventListener("fullscreenchange", syncResourcePreviewFullscreenButton);

if (els.resourcePreviewDialog) {
  els.resourcePreviewDialog.addEventListener("close", () => {
    els.resourcePreviewDialog.classList.remove("is-window-fullscreen");
    els.resourcePreviewFrame.removeAttribute("src");
    els.resourcePreviewFrame.hidden = false;
    resourcePreviewBaseUrl = "";
    resourcePreviewMode = "frame";
    resetResourcePreviewZoom();
    syncResourcePreviewFullscreenButton();
    if (els.resourcePreviewCode) {
      els.resourcePreviewCode.hidden = true;
      els.resourcePreviewCode.textContent = "";
    }
  });
  els.resourcePreviewDialog.addEventListener("click", (event) => {
    if (event.target === els.resourcePreviewDialog) closeResourcePreview();
  });
}

if (els.markdownFile && els.postForm) {
  els.markdownFile.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const text = await file.text();
    els.postForm.elements.content.value = text;
    if (!els.postForm.elements.title.value) {
      els.postForm.elements.title.value = file.name.replace(/\.md$/i, "");
    }
    renderPreview();
  });
}

if (els.postForm) {
  els.postForm.addEventListener("input", renderPreview);
  els.postForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const title = els.postForm.elements.title.value || "未命名文章";
    downloadText(`${slugify(title)}.md`, buildDraftMarkdown());
  });
}

if (els.downloadDraft && els.postForm) {
  els.downloadDraft.addEventListener("click", () => {
    const title = els.postForm.elements.title.value || "未命名文章";
    downloadText(`${slugify(title)}.md`, buildDraftMarkdown());
  });
}

if (els.githubLoginButton) {
  els.githubLoginButton.addEventListener("click", async () => {
    setWriterMessage(els.writerLoginMessage, "正在连接 GitHub...", "info");
    try {
      await verifyWriterToken(els.githubTokenInput.value);
    } catch (error) {
      updateWriterAuthView();
      setWriterMessage(els.writerLoginMessage, error.message, "error");
    }
  });
}

if (els.githubLogoutButton) {
  els.githubLogoutButton.addEventListener("click", () => {
    writerState = { token: "", user: null, files: [], currentPath: "", currentSha: "" };
    sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    if (els.githubTokenInput) els.githubTokenInput.value = "";
    updateWriterAuthView();
    setWriterMessage(els.writerLoginMessage, "已退出写作台。", "info");
    location.hash = "home";
  });
}

if (els.writerPostSelect) {
  els.writerPostSelect.addEventListener("change", async () => {
    setWriterMessage(els.writerSaveMessage, "", "info");
    try {
      await loadWriterPost(els.writerPostSelect.value);
    } catch (error) {
      setWriterMessage(els.writerSaveMessage, error.message, "error");
    }
  });
}

[els.writerTitle, els.writerDate, els.writerCategory, els.writerTags, els.writerPdf, els.writerPdfTitle, els.writerSummary, els.writerContent]
  .filter(Boolean)
  .forEach((input) => input.addEventListener("input", () => scheduleWriterPreview()));

if (els.writerPreviewRefresh) {
  els.writerPreviewRefresh.addEventListener("click", () => scheduleWriterPreview({ immediate: true }));
}

if (els.imagePreviewClose) {
  els.imagePreviewClose.addEventListener("click", closeImagePreview);
}

if (els.imagePreviewDialog) {
  els.imagePreviewDialog.addEventListener("close", () => {
    els.imagePreviewImage?.removeAttribute("src");
  });
  els.imagePreviewDialog.addEventListener("click", (event) => {
    if (event.target === els.imagePreviewDialog) closeImagePreview();
  });
}

if (els.writerNewButton) {
  els.writerNewButton.addEventListener("click", () => {
    resetWriterForm();
    setWriterMessage(els.writerSaveMessage, "已切换到新建文章。", "info");
  });
}

if (els.writerSaveButton) {
  els.writerSaveButton.addEventListener("click", async () => {
    setWriterMessage(els.writerSaveMessage, "正在保存到 GitHub...", "info");
    els.writerSaveButton.disabled = true;
    try {
      await saveWriterPost();
    } catch (error) {
      setWriterMessage(els.writerSaveMessage, error.message, "error");
    } finally {
      els.writerSaveButton.disabled = false;
    }
  });
}

if (els.writerContent) {
  els.writerContent.addEventListener("paste", async (event) => {
    const imageFiles = [...(event.clipboardData?.items || [])]
      .filter((item) => item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter(Boolean);
    if (imageFiles.length && writerState.user?.login === GITHUB_OWNER) {
      event.preventDefault();
      try {
        await importWriterImages(imageFiles, "pasted");
      } catch (error) {
        setWriterMessage(els.writerSaveMessage, error.message, "error");
      }
      return;
    }

    const pastedText = event.clipboardData?.getData("text/plain") || "";
    if (hasObsidianImageReferences(pastedText)) {
      setTimeout(async () => {
        scheduleWriterPreview();
        try {
          if (obsidianDirectoryHandle && await hasObsidianDirectoryPermission(obsidianDirectoryHandle)) {
            await syncObsidianImagesFromDirectory();
          } else {
            setWriterMessage(els.writerSaveMessage, "已识别 Obsidian 图片链接。关联一次图片文件夹后即可自动批量上传。", "info");
          }
        } catch (error) {
          setWriterMessage(els.writerSaveMessage, `自动同步图片失败：${error.message}`, "error");
        }
      });
    }
  });
}

if (els.writerObsidianImagesButton && els.writerObsidianImagesInput) {
  els.writerObsidianImagesButton.addEventListener("click", async () => {
    els.writerObsidianImagesButton.disabled = true;
    try {
      await connectObsidianDirectory();
    } catch (error) {
      if (error?.name !== "AbortError") {
        setWriterMessage(els.writerSaveMessage, `关联附件文件夹失败：${error.message}`, "error");
      }
    } finally {
      els.writerObsidianImagesButton.disabled = false;
    }
  });
  els.writerObsidianImagesInput.addEventListener("change", async () => {
    els.writerObsidianImagesButton.disabled = true;
    try {
      await importObsidianFolderFallback(els.writerObsidianImagesInput.files);
    } catch (error) {
      setWriterMessage(els.writerSaveMessage, error.message, "error");
    } finally {
      els.writerObsidianImagesInput.value = "";
      els.writerObsidianImagesButton.disabled = false;
    }
  });
}

loadSiteData().then(async () => {
  renderAll();
  await restoreObsidianDirectory();
  if (writerState.token) {
    try {
      await verifyWriterToken(writerState.token, true);
    } catch (error) {
      writerState = { token: "", user: null, files: [], currentPath: "", currentSha: "" };
      sessionStorage.removeItem(TOKEN_STORAGE_KEY);
      setWriterMessage(els.writerLoginMessage, error.message, "error");
      updateWriterAuthView();
    }
  }
  setRoute(location.hash.replace("#", ""));
});
