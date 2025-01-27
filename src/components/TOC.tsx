import React, { useEffect, useState } from "react";
import matter from "gray-matter";
import styles from "./TOC.module.scss";
import { HashLink as Link } from "react-router-hash-link";

type TOCProps = {
  fileName: string;
};

const TOC: React.FC<TOCProps> = ({ fileName }) => {
  const [headings, setHeadings] = useState<
    { text: string; id: string; level: number }[]
  >([]);

  useEffect(() => {
    const fetchContent = async () => {
      if (fileName) {
        const fileModules = import.meta.glob("/posts/**/*.md", {
          query: "?raw",
          import: "default",
        });
        const filePath = `/posts/${fileName}.md`;

        if (fileModules[filePath]) {
          const content = await fileModules[filePath]();
          const { content: mdContent } = matter(content as string);

          const headingRegex = /^(?!#\s)(#{1,5})\s+(.*)$/gm;
          const matches = [...mdContent.matchAll(headingRegex)];

          const extractedHeadings = matches.map((match) => ({
            text: match[2],
            id: match[2].toLowerCase().replace(/\s+/g, "-"),
            level: match[1].length,
          }));

          setHeadings(extractedHeadings);
        } else {
          console.error(`File not found: ${filePath}`);
        }
      }
    };

    fetchContent();
  }, [fileName]);

  return (
    <ul className={styles.toc_list}>
      {headings.map((heading, index) => (
        <li key={index} style={{ marginLeft: (heading.level - 1) * 10 }}>
          <Link to={`#${heading.id}`}>{heading.text}</Link>
        </li>
      ))}
    </ul>
  );
};

export default TOC;
