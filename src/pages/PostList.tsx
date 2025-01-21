import React, { useEffect, useState } from 'react';
import matter from 'gray-matter';

interface Post {
  title: string;
  date: string;
  fileName: string;
}

interface PostListProps {
  onSelect: (fileName: string) => void;
}

const PostList: React.FC<PostListProps> = ({ onSelect }) => {
  const [posts, setPosts] = useState<Post[]>([]);

  useEffect(() => {
    const importMarkdownFiles = async () => {
      const files = import.meta.glob('/posts/*.md', { query: '?raw', import: 'default' });
      const postList: Post[] = [];

      for (const path in files) {
        const content = await files[path]();
        const { data } = matter(content);
        const fileName = path.split('/').pop() || '';
        const date = fileName.split('-').slice(0, 3).join('-');
        postList.push({ title: data.title, date, fileName });
      }
      postList.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

      setPosts(postList);
    };

    importMarkdownFiles();
  }, []);
  
  return (
    <div className="sidebar">
      <p className="sidebar-title"> 블로그 리스트 </p>
      <p className="sidebar-count">전체 블로그 {posts.length}개</p>
      <ul className="sidebar-list">
        {posts.map((post, index) => (
          <li key={index} onClick={() => onSelect(post.fileName)}>
            {post.title} ({post.date})
          </li>
        ))}
      </ul>
    </div>
  );
};

export default PostList;