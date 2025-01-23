import React, { useEffect, useState } from 'react';
import './TOC.css';

interface TOCProps {
  fileName: string;
}

const TOC: React.FC<TOCProps> = ({ fileName }) => {
  const [headings, setHeadings] = useState<{ text: string; id: string; level: number }[]>([]);

  useEffect(() => {
    const extractHeadings = () => {
      const contentElement = document.querySelector('.markdown-content');
      if (contentElement) {
        const headingElements = contentElement.querySelectorAll('h1, h2, h3, h4, h5');
        const extractedHeadings = Array.from(headingElements).map(heading => ({
          text: heading.textContent || '',
          id: heading.id,
          level: parseInt(heading.tagName.substring(1), 10),
        }));
        setHeadings(extractedHeadings);
      }
    };

    extractHeadings();
  }, [fileName]);

  return (
    <div className='toc'>
      <ul className='toc-list'>
        {headings.map((heading, index) => (
          <li key={index} style={{ marginLeft: (heading.level - 1) * 10 }}>
            <a href={`#${heading.id}`}>{heading.text}</a>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TOC;