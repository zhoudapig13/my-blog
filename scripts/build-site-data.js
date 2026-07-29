const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const postsDir = path.join(root, "content", "posts");
const diaryDir = path.join(root, "content", "diary");
const dataDir = path.join(root, "data");
const outputPath = path.join(dataDir, "site.json");

function readJson(relativePath, fallback) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath)) return fallback;
  return JSON.parse(fs.readFileSync(fullPath, "utf8"));
}

function parseFrontmatter(source, fileName) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) {
    return {
      meta: {
        title: fileName.replace(/\.md$/i, ""),
        category: "未分类",
        tags: [],
        date: new Date().toISOString().slice(0, 10),
        pdf: ""
      },
      content: source
    };
  }

  const meta = {};
  const lines = match[1].split(/\r?\n/);
  let currentKey = null;

  for (const line of lines) {
    const listItem = line.match(/^\s*-\s+["']?(.+?)["']?\s*$/);
    if (listItem && currentKey) {
      meta[currentKey] = Array.isArray(meta[currentKey]) ? meta[currentKey] : [];
      meta[currentKey].push(listItem[1]);
      continue;
    }

    const pair = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!pair) continue;
    currentKey = pair[1];
    const value = pair[2].trim();
    if (!value) {
      meta[currentKey] = [];
      continue;
    }
    meta[currentKey] = value.replace(/^["']|["']$/g, "");
  }

  return { meta, content: match[2].trim() };
}

function excerpt(content, length = 140) {
  const plain = content
    .replace(/```[\s\S]*?```/g, "")
    .replace(/[#>*_`\-[\]()]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return plain.length > length ? `${plain.slice(0, length)}...` : plain;
}

function loadMarkdownCollection(collectionDir) {
  if (!fs.existsSync(collectionDir)) return [];
  return fs
    .readdirSync(collectionDir)
    .filter((file) => file.endsWith(".md"))
    .map((file) => {
      const source = fs.readFileSync(path.join(collectionDir, file), "utf8");
      const { meta, content } = parseFrontmatter(source, file);
      return {
        id: file.replace(/\.md$/i, ""),
        title: meta.title || file.replace(/\.md$/i, ""),
        category: Array.isArray(meta.category) ? meta.category[0] || "未分类" : meta.category || "未分类",
        tags: Array.isArray(meta.tags) ? meta.tags : [],
        date: meta.date || new Date().toISOString().slice(0, 10),
        pdf: meta.pdf || "",
        pdfTitle: meta.pdfTitle || "",
        content,
        excerpt: excerpt(content)
      };
    })
    .sort((a, b) => b.date.localeCompare(a.date));
}

const siteData = {
  profile: readJson("data/profile.json", {}),
  resources: readJson("data/resources.json", { collections: [] }).collections || [],
  friends: readJson("data/friends.json", { items: [] }).items || [],
  plans: readJson("data/plans.json", { items: [] }).items || [],
  posts: loadMarkdownCollection(postsDir),
  diary: loadMarkdownCollection(diaryDir)
};

fs.mkdirSync(dataDir, { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(siteData, null, 2)}\n`);
console.log(
  `Generated ${path.relative(root, outputPath)} with ${siteData.posts.length} posts and ${siteData.diary.length} diary entries.`
);
