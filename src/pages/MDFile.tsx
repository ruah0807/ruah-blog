import React, { useEffect, useState, useRef } from 'react';
import matter from 'gray-matter';
import Markdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/cjs/styles/prism";
import './MDFile.css';
import TOC from '../components/TOC';

interface MDFileProps {
  fileName: string;
}

interface ExtraProps {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

const MDFile: React.FC<MDFileProps> = ({  fileName }) => {
  const [content, setContent] = useState<string>('');
  const [title, setTitle] = useState<string>('');
  const [date, setDate] = useState<string>('');
  const contentRef = useRef<HTMLDivElement>(null);
  const tocRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
      const fetchContent = async () => {
        if (fileName) {
          const fileModules = import.meta.glob('/posts/**/*.md', { query: '?raw', import: 'default' });
          const filePath = `/posts/${fileName}`;
    
          if (fileModules[filePath]) {
            const content = await fileModules[filePath]();
            const { data, content: mdContent } = matter(content as string);
            setContent(mdContent);
            setTitle(data.title || '');
    
            const date = fileName.split('-').slice(0, 3).join('-');
            setDate(date);
            console.log(`fileName : ${fileName}`);
            console.log(`filePath : ${filePath}`);
            console.log(`제목 : ${data.title}, 날짜 : ${date}`);
          } else {
            console.error(`File not found: ${filePath}`);
          }
        }
      };
    fetchContent();
  }, [fileName]);

  useEffect(() => {
    const handleScroll = () => {
      if (contentRef.current && tocRef.current) {
        tocRef.current.scrollTop = contentRef.current.scrollTop;
      }
    };

    const contentElement = contentRef.current;
    contentElement?.addEventListener('scroll', handleScroll);

    return () => {
      contentElement?.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <div className='markdown-container'>
      <div className='markdown-content' ref={contentRef}>
        <div className='md-header'>
          <div className='md-title'>{title}</div>
          <div className='md-date'>{date}</div>
        </div>
        <div className='line'></div>
        <Markdown
          components={{
            h1: ({ ...props }) => {
                const text = props.children && typeof props.children === 'string' ? props.children : '';
                return <h1 id={text.toLowerCase().replace(/\s+/g, '-')}>{props.children}</h1>;
              },
            h2: ({ ...props }) => {
                const text = props.children && typeof props.children === 'string' ? props.children : '';
                return <h2 id={text.toLowerCase().replace(/\s+/g, '-')}>{props.children}</h2>;
              },
            h3: ({ ...props }) => {
                const text = props.children && typeof props.children === 'string' ? props.children : '';
                return <h3 id={text.toLowerCase().replace(/\s+/g, '-')}>{props.children}</h3>;
              },
            h4: ({ ...props }) => {
                const text = props.children && typeof props.children === 'string' ? props.children : '';
                return <h4 id={text.toLowerCase().replace(/\s+/g, '-')}>{props.children}</h4>;
              },
            h5: ({ ...props }) => {
                const text = props.children && typeof props.children === 'string' ? props.children : '';
                return <h5 id={text.toLowerCase().replace(/\s+/g, '-')}>{props.children}</h5>;
              },            code({ inline, className, children, ...props }: ExtraProps) {
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
      <div className='toc-container' ref={tocRef}>
        <TOC fileName={fileName} />
      </div>
    </div>
  );
};

export default MDFile;