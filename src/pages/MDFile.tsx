import React, { useEffect, useState } from 'react';
import matter from 'gray-matter';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { solarizedlight } from 'react-syntax-highlighter/dist/esm/styles/prism';
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
      <ReactMarkdown
        components={{
          code({ inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={solarizedlight}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
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
      </ReactMarkdown>
    </div>
  );
};

export default MDFile;