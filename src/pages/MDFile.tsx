import React, { useEffect, useState } from 'react';
import matter from 'gray-matter';
import Markdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/cjs/styles/prism";
import './MDFile.css';

interface MDFileProps {
  fileName: string;
}

// ExtraProps 타입 정의
interface ExtraProps {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
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
          code({ inline, className, children, ...props }: ExtraProps) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={materialDark}
                language={match[1]}
                PreTag="div"
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