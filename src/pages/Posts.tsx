import React, { useEffect, useState } from 'react';
import { remark } from 'remark';
import html from 'remark-html';
import post from '../../posts/workflow_1.md';

const Posts: React.FC = () => {
  const [content, setContent] = useState<string>('');
  

  useEffect(() => {
    const fetchMarkdown = async () => {
      const file = await import(post);
      const response = await fetch(file.default);
      const text = await response.text();
      const result = await remark().use(html).process(text);
      setContent(result.toString());
    };

    fetchMarkdown();
  }, []);

  return (
    <div>
      <h2>Posts</h2>
      <div dangerouslySetInnerHTML={{ __html: content }} />
    </div>
  );
};

export default Posts;
