import React, { useEffect, useState } from 'react';
import PostCard from '../components/PostCard';
import matter from 'gray-matter';

interface Post {
  title: string;
  date: string;
}

const Home: React.FC = () => {
  const [posts, setPosts] = useState<Post[]>([]);

  useEffect(() => {
    const importMarkdownFiles = async () => {
      const files = import.meta.glob('/posts/*.md', { as: 'raw' });
      const postList: Post[] = [];

      for (const path in files) {
        const content = await files[path]();
        const { data } = matter(content);
        const fileName = path.split('/').pop() || '';
        const date = fileName.split('-').slice(0, 3).join('-');
        postList.push({ title: data.title, date });
      }
      console.log(postList);
      setPosts(postList);
    };

    importMarkdownFiles();
  }, []);

  return (
    <div className="page-content">
      <h2 className='page-title'>Posts List</h2>
      <ul>
        {posts.map((post, index) => (
          <li key={index}>
            <PostCard title={post.title} date={post.date} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Home;