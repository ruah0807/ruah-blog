import React, { useEffect, useState } from 'react';
import matter from 'gray-matter';
import { useNavigate } from 'react-router-dom';

interface Post {
  title: string;
  date: string;
  fileName: string;
  subtitle: string;
}

interface PostListProps {
  onSelect: (fileName: string) => void;
}

const PostList: React.FC<PostListProps> = ({ onSelect }) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const navigate = useNavigate();
  const [selectedPost, setSelectedPost] = useState<string>('');

  useEffect(() => {
    const importMarkdownFiles = async () => {
      const files = import.meta.glob('/posts/*.md', { query: '?raw', import: 'default' });
      const postList: Post[] = [];

      for (const path in files) {
        const content = await files[path]();
        const { data } = matter(content);
        const fileName = path.split('/').pop() || '';
        const date = fileName.split('-').slice(0, 3).join('-');
        const subtitle = fileName.split('-').slice(3).join('-').replace('.md', '');
        postList.push({ title: data.title, date, fileName, subtitle });
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
          <li key={index} onClick={() => {
            onSelect(post.fileName);
            navigate(`/${post.subtitle}`);
            setSelectedPost(post.fileName);
          }}
          className={selectedPost === post.fileName ? 'active' : ''}>
            {post.title}
            <span className='sidebar-date'> {post.date}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default PostList;