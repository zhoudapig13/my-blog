let state = {
  theme: "journal",
  profile: {},
  posts: [],
  resources: [],
  plans: []
};

const FILTER_ALL = "全部";
const previewTheme = new URLSearchParams(location.search).get("theme");
let activeFilter = FILTER_ALL;

const routeMap = {
  home: document.querySelector("#homeView"),
  blog: document.querySelector("#blogView"),
  post: document.querySelector("#postView"),
  resources: document.querySelector("#resourcesView"),
  plan: document.querySelector("#planView"),
  studio: document.querySelector("#studioView")
};

const els = {
  body: document.body,
  search: document.querySelector("#searchInput"),
  profileImage: document.querySelector("#profileImage"),
  profileName: document.querySelector("#profileName"),
  profileBio: document.querySelector("#profileBio"),
  recentPosts: document.querySelector("#recentPosts"),
  postList: document.querySelector("#postList"),
  postReader: document.querySelector("#postReader"),
  categoryList: document.querySelector("#categoryList"),
  tagList: document.querySelector("#tagList"),
  categoryCount: document.querySelector("#categoryCount"),
  tagCount: document.querySelector("#tagCount"),
  blogFilters: document.querySelector("#blogFilters"),
  resourceList: document.querySelector("#resourceList"),
  planList: document.querySelector("#planList"),
  progressValue: document.querySelector("#progressValue"),
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
      theme: ["atelier", "journal", "garden"].includes(previewTheme) ? previewTheme : state.theme
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

  text = text.replace(/```([\s\S]*?)```/g, (_, code) => {
    const token = `@@BLOCK${blocks.length}@@`;
    blocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
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
  let listOpen = false;

  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };

  for (const rawLine of lines) {
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

    if (/^#{1,4}\s/.test(line)) {
      closeList();
      const level = line.match(/^#+/)[0].length;
      html.push(`<h${level}>${inlineMarkdown(line.replace(/^#{1,4}\s/, ""))}</h${level}>`);
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

function inlineMarkdown(value) {
  return value
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[\[([^\]]+)\]\]/g, '<span class="wiki-link">$1</span>')
    .replace(/\$([^$\n]+)\$/g, (_, math) => `<span class="math-inline">\\(${cleanMath(math)}\\)</span>`)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function cleanMath(value) {
  return String(value).trim().replace(/\\([*_])/g, "$1");
}

function typesetMath(container = document.body) {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise([container]).catch((error) => console.error(error));
  }
}

function getSearchTerm() {
  return els.search.value.trim().toLowerCase();
}

function getFilteredPosts() {
  const term = getSearchTerm();
  return [...state.posts]
    .sort((a, b) => b.date.localeCompare(a.date))
    .filter((post) => {
      const haystack = [
        post.title,
        post.category,
        post.tags.join(" "),
        post.content,
        post.pdf
      ].join(" ").toLowerCase();
      const matchesSearch = !term || haystack.includes(term);
      const matchesFilter =
        activeFilter === FILTER_ALL ||
        post.category === activeFilter ||
        post.tags.includes(activeFilter);
      return matchesSearch && matchesFilter;
    });
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

function createPostCard(post) {
  const template = document.querySelector("#postCardTemplate").content.cloneNode(true);
  const card = template.querySelector(".post-card");
  card.querySelector(".post-meta").textContent = `${post.date} / ${post.category || "未分类"}`;
  card.querySelector("h3").textContent = post.title;
  card.querySelector("p").textContent = post.excerpt || excerpt(post.content);
  card.querySelector(".post-tags").innerHTML = post.tags
    .map((tag) => `<a href="#blog" data-filter="${escapeHtml(tag)}">${escapeHtml(tag)}</a>`)
    .join("");
  card.querySelector("button").addEventListener("click", () => openPost(post.id));
  card.querySelector("h3").addEventListener("click", () => openPost(post.id));
  return card;
}

function renderPosts() {
  const posts = getFilteredPosts();
  els.recentPosts.innerHTML = "";
  state.posts.slice(0, 4).forEach((post) => els.recentPosts.append(createPostCard(post)));

  els.postList.innerHTML = "";
  if (!posts.length) {
    els.postList.innerHTML = '<div class="empty-state">没有匹配的文章。换个关键词试试，或者去 GitHub CMS 发布一篇新的。</div>';
    return;
  }

  posts.forEach((post) => els.postList.append(createPostCard(post)));
}

function openPost(id) {
  location.hash = `post/${encodeURIComponent(id)}`;
}

function renderPostPage(id) {
  const post = state.posts.find((item) => item.id === decodeURIComponent(id || ""));
  if (!post) {
    els.postReader.innerHTML = '<div class="empty-state">没有找到这篇文章。<a href="#blog">返回博客列表</a></div>';
    return;
  }

  const pdf = post.pdf
    ? `<p><strong>PDF：</strong><a href="${escapeHtml(post.pdf)}" target="_blank" rel="noreferrer">${escapeHtml(post.pdf)}</a></p>`
    : "";

  els.postReader.innerHTML = `
    <a class="back-link" href="#blog">返回文章列表</a>
    <p class="eyebrow">${escapeHtml(post.category || "未分类")} / ${escapeHtml(post.date)}</p>
    ${pdf}
    ${markdownToHtml(post.content)}
  `;
  typesetMath(els.postReader);
}

function renderResources() {
  const term = getSearchTerm();
  const resources = state.resources.filter((resource) =>
    [resource.title, resource.type, resource.url, resource.description].join(" ").toLowerCase().includes(term)
  );
  els.resourceList.innerHTML =
    resources
      .map(
        (resource) => `
          <article class="resource-card">
            <div class="resource-meta"><span>${escapeHtml(resource.type)}</span></div>
            <h3>${escapeHtml(resource.title)}</h3>
            <p>${escapeHtml(resource.description || "暂无说明")}</p>
            ${
              resource.url
                ? `<a href="${escapeHtml(resource.url)}" target="_blank" rel="noreferrer">打开资源</a>`
                : '<span class="empty-state">尚未填写链接</span>'
            }
          </article>
        `
      )
      .join("") || '<div class="empty-state">没有匹配的资源。</div>';
}

function renderPlans() {
  const done = state.plans.filter((plan) => plan.status === "done").length;
  const progress = state.plans.length ? Math.round((done / state.plans.length) * 100) : 0;
  els.progressValue.textContent = `${progress}%`;
  document.querySelector(".progress-ring").style.setProperty("--progress", `${progress}%`);

  const statusLabel = {
    todo: "未开始",
    doing: "进行中",
    done: "已完成"
  };

  els.planList.innerHTML =
    state.plans
      .map(
        (plan) => `
          <article class="plan-item">
            <div>
              <strong>${escapeHtml(plan.goal)}</strong>
              <span class="status">${statusLabel[plan.status] || "未开始"}</span>
            </div>
          </article>
        `
      )
      .join("") || '<div class="empty-state">还没有计划。去 GitHub CMS 添加一个目标，进度会自动更新。</div>';
}

function renderPreview() {
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
  renderResources();
  renderPlans();
  renderPreview();
}

function setRoute(rawRoute) {
  const [route, id] = (rawRoute || "home").split("/");
  const view = routeMap[route] ? route : "home";

  Object.entries(routeMap).forEach(([key, element]) => {
    element.classList.toggle("active", key === view);
  });
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("active", link.dataset.route === view);
  });

  if (view === "post") {
    renderPostPage(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
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
  const content = data.get("content").trim() || `# ${title}\n\n正文待补充。`;
  return `---\ntitle: "${title.replaceAll('"', '\\"')}"\ncategory: "${category.replaceAll('"', '\\"')}"\ntags:\n${tags.map((tag) => `  - "${tag.replaceAll('"', '\\"')}"`).join("\n") || "  []"}\ndate: "${new Date().toISOString().slice(0, 10)}"\npdf: "${pdf.replaceAll('"', '\\"')}"\n---\n\n${content}\n`;
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

document.addEventListener("click", (event) => {
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

window.addEventListener("hashchange", () => setRoute(location.hash.replace("#", "")));

els.search.addEventListener("input", () => {
  renderPosts();
  renderResources();
});

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

els.postForm.addEventListener("input", renderPreview);

els.postForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const title = els.postForm.elements.title.value || "未命名文章";
  downloadText(`${slugify(title)}.md`, buildDraftMarkdown());
});

els.downloadDraft.addEventListener("click", () => {
  const title = els.postForm.elements.title.value || "未命名文章";
  downloadText(`${slugify(title)}.md`, buildDraftMarkdown());
});

loadSiteData().then(() => {
  renderAll();
  setRoute(location.hash.replace("#", ""));
});
