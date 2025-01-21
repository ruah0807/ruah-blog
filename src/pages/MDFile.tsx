import React, { useEffect, useState } from 'react';
import matter from 'gray-matter';
import Markdown from "react-markdown";
import { materialDark } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

import './MDFile.css';

interface MDFileProps {
  fileName: string;
}

const MDFile: React.FC<MDFileProps> = ({ fileName }) => {
  const [content, setContent] = useState<string>('');

  useEffect(() => {
    const fetchContent = async () => {
      if (fileName) {
        const filePath = `/posts/${fileName}`;
        const fileModules = import.meta.glob('/posts/*.md', { query: '?raw', import: 'default' });
        const content = await fileModules[filePath]();
        const { content: mdContent } = matter(content);
        setContent(mdContent);
        console.log(mdContent);
      }
    };

    fetchContent();
  }, [fileName]);

  return (
    <div className='markdown-content'>
      <Markdown
      components={{
        code({ inline, className = "", children, ...props }: { inline: boolean, className?: string, children: React.ReactNode }) {
          const match = /language-(\w+)/.exec(className);

          return !inline && match ? (
            <SyntaxHighlighter
              style={materialDark}
              PreTag="div"
              language={match[1]}
              {...props}
            >
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        }
      }}
    >
      {content}
    </Markdown>
    </div>
  );
};

export default MDFile;