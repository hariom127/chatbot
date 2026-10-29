// Minimal Markdown for bot answers: headings, bullet/numbered lists, **bold**, paragraphs.
// Builds React elements only (no innerHTML), so answer text can never inject markup.

function inline(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4
      ? <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>
      : part
  );
}

const HEADING_CLASSES = {
  1: "mb-2 text-lg font-bold text-gray-900",
  2: "mb-1.5 mt-4 text-base font-semibold text-gray-900",
  3: "mb-1 mt-3 font-semibold text-gray-900",
};

export default function Markdown({ text }) {
  const blocks = [];
  let list = null;
  let para = [];

  const flushPara = () => {
    if (para.length) blocks.push(<p key={blocks.length} className="my-1.5">{inline(para.join(" "))}</p>);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={blocks.length} className={`my-1.5 space-y-1 pl-5 ${list.ordered ? "list-decimal" : "list-disc"}`}>
        {list.items.map((item, i) => <li key={i}>{inline(item)}</li>)}
      </Tag>
    );
    list = null;
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);

    if (!line) {
      flushPara();
      flushList();
    } else if (heading) {
      flushPara();
      flushList();
      const level = Math.min(heading[1].length, 3);
      const Tag = `h${level + 2}`;
      blocks.push(<Tag key={blocks.length} className={HEADING_CLASSES[level]}>{inline(heading[2])}</Tag>);
    } else if (bullet || numbered) {
      flushPara();
      const ordered = Boolean(numbered);
      if (list && list.ordered !== ordered) flushList();
      list = list || { ordered, items: [] };
      list.items.push((bullet || numbered)[1]);
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();

  return <div className="[&>*:first-child]:mt-0">{blocks}</div>;
}
