import React, { useEffect, useState } from 'react';
import matter from 'gray-matter';
import { useNavigate } from 'react-router-dom';

interface Post {
  title: string;
  date: string;
  fileName: string;
  subtitle: string;
  category: string | null;
}

interface PostListProps {
  onSelect: (fileName: string) => void;
}

const PostList: React.FC<PostListProps> = ({ onSelect }) => {
  const [state, setState] = useState({
    postsByCategory: {} as Record<string, Post[]>,
    selectedCategory: null as string | null,
    selectedPost: ''
  });
  const navigate = useNavigate();

  useEffect(() => {
    const importMarkdownFiles = async () => {
      const files = import.meta.glob('/posts/**/*.md', { query: '?raw', import: 'default' });
      const postList: Post[] = [];

      for (const path in files) {
        const content = await files[path]();
        const { data } = matter(content as string);
        const parts = path.split('/');
        const fileName = parts.slice(2).join('/'); // Include category if present
        const category = parts.length > 3 ? parts[2] : null;
        const date = fileName.split('-').slice(0, 3).join('-');
        const subtitle = category ? `${category}/${fileName.split('-').slice(3).join('-').replace('.md', '')}` : fileName.split('-').slice(3).join('-').replace('.md', '');
        postList.push({ title: data.title, date, fileName, subtitle, category });
      } 
      const groupedPosts = postList.reduce((acc, post) => {
        const key = post.category || 'Others';
        if (!acc[key]) {
          acc[key] = [];
        }
        acc[key].push(post);
        return acc;
      }, {} as Record<string, Post[]>);

      setState(preState => ({ ...preState, postsByCategory: groupedPosts }));
    };

    importMarkdownFiles();
  }, []);
  
  return (
    <div className="sidebar">
      <p className="sidebar-title"> 블로그 리스트 </p>
      <p className="sidebar-count">전체 블로그 {Object.values(state.postsByCategory).flat().length}개</p>
      {Object.entries(state.postsByCategory).sort(([a], [b]) => a === 'Others' ? 1 : b === 'Others' ? -1 : 0).map(([category, posts]) => (
        <div className='sidebar-category' key={category}>
          {category !== 'Others' && (
            <div className='sidebar-folder' draggable="false" onClick={() => setState(prevState => ({
              ...prevState,
              selectedCategory: prevState.selectedCategory === category ? null : category
            }))}>
              {state.selectedCategory === category ? '▼ ' : '▷ '} {category}
            </div>
          )}
          {(state.selectedCategory === category || category === 'Others') && (
            <ul className="sidebar-list">
              {posts.map((post, index) => (
                <li key={index} onClick={() => {
                  onSelect(post.fileName);
                  navigate(`/${post.subtitle}`);
                  setState(prevState => ({ ...prevState, selectedPost: post.fileName }));
                }}
                className={`file-item ${state.selectedPost === post.fileName ? 'active' : ''}`}>
                  {post.title}
                  <span className='sidebar-date'> {post.date}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
};

export default PostList;