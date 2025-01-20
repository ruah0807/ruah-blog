import React from 'react';
import './PostCard.css'; // 스타일을 위한 CSS 파일

interface PostCardProps {
  title: string;
  date: string;
}

const PostCard: React.FC<PostCardProps> = ({ title, date }) => {
  return (
    <div className="post-card">
      <h3>{title}</h3>
      <p>{date}</p>
    </div>
  );
};

export default PostCard; 